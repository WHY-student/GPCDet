import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Mish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1,g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.conv = Conv(c1, c_, k, s, None, g, act)
        self.cv1 = Conv(c_, c_//2, 5, 1, None, c_//2, act)
        self.cv2 = Conv(c_, c_//2, 3, 1, None, c_//2, act)

    #improve1
    def forward(self, x):
        x1 = self.conv(x)
        x2_3=self.cv1(x1)
        x2_5=self.cv2(x1)
        x3 = torch.cat((x1,x2_3,x2_5), 1)
        # shuffle

        b, n, h, w = x3.data.size()
        b_n = b * n // 2
        y = x3.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        y=torch.cat((y[0], y[1]), 1)
        return y


# class GSConv(nn.Module):
#     def __init__(self, c1, c2, k=1, s=1,g=1, act=True):
#         super().__init__()
#         self.l_conv = ConvModule(c1,c2,1,inplace=False)
#         self.conv1_3=nn.Conv2d(in_channels=c2,out_channels=c2//2,kernel_size=3,stride=1,padding=1)
#         self.conv1_5 = nn.Conv2d(in_channels=c2, out_channels=c2 // 2, kernel_size=5, stride=1, padding=2)
#         self.conv2_3_2=nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=2,dilation=2)
#         self.conv2_3_3 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=3,dilation=3)
#     #improve1
#     def forward(self, x):
#         x=self.l_conv(x)
#         y1=self.conv1_3(x)
#         y2=self.conv1_5(x)
#         y=torch.cat((y1,y2),dim=1)
#         z1=self.conv2_3_2(y)
#         z2=self.conv2_3_3(y)
#         return x+z1+z2
