import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F
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


# class GSConv(nn.Module):
#     # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=1, s=1,g=1, act=True):
#         super().__init__()
#         c_ = c2 // 2
#         self.conv = Conv(c1, c_, k, s, None, g, act)
#         self.cv1 = Conv(c_, c_//2, 5, 1, None, c_//2, act)
#         self.cv2 = Conv(c_, c_//2, 3, 1, None, c_//2, act)
#
#     #improve1
#     def forward(self, x):
#         x1 = self.conv(x)
#         x2_3=self.cv1(x1)
#         x2_5=self.cv2(x1)
#         x3 = torch.cat((x1,x2_3,x2_5), 1)
#         # shuffle
#
#         b, n, h, w = x3.data.size()
#         b_n = b * n // 2
#         y = x3.reshape(b_n, 2, h * w)
#         y = y.permute(1, 0, 2)
#         y = y.reshape(2, -1, n // 2, h, w)
#         y=torch.cat((y[0], y[1]), 1)
#         return y

class SCConv(nn.Module):
    def __init__(self, inplanes, planes,k=3,groups=1, pooling_r=4):
        super().__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    Conv(inplanes, planes, k=k, s=1,g=groups),
                    )
        self.k3 = nn.Sequential(
                    Conv(inplanes, planes, k=k, s=1,g=groups),
                  #  norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
            Conv(inplanes, planes, k=k, s=1, g=groups),
                 #   norm_layer(planes),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4
        return out

# class GSConv(nn.Module):
#     """SCNet SCBottleneck
#     """
#
#     def __init__(self, c1, c2, stride=1,dilation=1,groups=1):
#         super().__init__()
#         group_width = (int)(c2 /2)
#         self.conv1_a = nn.Conv2d(c1, group_width, kernel_size=1, bias=False)
#         #self.bn1_a = norm_layer(group_width)
#         self.conv1_b = nn.Conv2d(c1, group_width, kernel_size=1, bias=False)
#        # self.bn1_b = norm_layer(group_width)
#         self.k1 = nn.Sequential(
#                     nn.Conv2d(
#                         group_width, group_width, kernel_size=3, stride=stride,
#                         padding=dilation, dilation=dilation,
#                         groups=groups, bias=False),
#                     #norm_layer(group_width),
#                     )
#
#         self.scconv = SCConv(
#             group_width, group_width, stride=stride,
#             padding=dilation, dilation=dilation,
#             groups=groups, pooling_r=4)
#
#         self.conv3 = nn.Conv2d(
#             c2, c2, kernel_size=1, bias=False)
#         #self.bn3 = norm_layer(planes*4)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.dilation = dilation
#         self.stride = stride
#
#     def forward(self, x):
#
#         out_a= self.conv1_a(x)
#         #out_a = self.bn1_a(out_a)
#         out_b = self.conv1_b(x)
#         #out_b = self.bn1_b(out_b)
#         out_a = self.relu(out_a)
#         out_b = self.relu(out_b)
#
#         out_a = self.k1(out_a)
#         out_b = self.scconv(out_b)
#         out_a = self.relu(out_a)
#         out_b = self.relu(out_b)
#
#
#         out = self.conv3(torch.cat([out_a, out_b], dim=1))
#        # out = self.bn3(out)
#         #out += residual
#        # out = self.relu(out)
#
#         return out
# class GSConv(nn.Module):
#     """SCNet SCBottleneck
#     """
#
#     def __init__(self, c1, c2, stride=1,dilation=1,groups=1):
#         super().__init__()
#         self.conv= nn.Sequential(
#             Conv(c1, c2, k=1, s=1, g=groups)
#         )
#         group_width = (int)(c2 /2)
#         self.k1 = nn.Sequential(
#             Conv(group_width, group_width, k=3, s=1, g=groups))
#
#         self.scconv = SCConv(
#             group_width, group_width,3,groups, 4)
#
#     def forward(self, x):
#         x=self.conv(x)
#         b, c, h, w = x.data.size()
#         x1, x2 = torch.split(x, c // 2, dim=1)
#         out_a = self.k1(x1)
#         out_b = self.scconv(x2)
#         out =torch.cat([out_a, out_b], dim=1)
#         return out

# class GSConv(nn.Module):
#     """SCNet SCBottleneck
#     """
#
#     def __init__(self, c1, c2, stride=1,dilation=1,groups=1):
#         super().__init__()
#         # self.conv1= nn.Sequential(
#         #     Conv(c1, c2//2, k=1, s=1, g=groups)
#         # )
#         self.conv1 = nn.Sequential(
#             Conv(c1, c2, k=1, s=1, g=groups)
#         )
#         self.conv2 = nn.Sequential(
#             Conv(c2//2, c2 // 4, k=3, s=1, g=groups)
#         )
#         self.conv3 = nn.Sequential(
#             Conv(c2//2, c2 // 4, k=5, s=1, g=groups)
#         )
#         self.k1_1= nn.Sequential(
#             Conv(c2//4, c2//4, k=1, s=1, g=groups))
#         self.scconv1 = SCConv(
#             inplanes=c2//4, planes=c2//4,k=1,groups=groups,pooling_r=4)
#         self.k1_2  = nn.Sequential(
#             Conv(c2//8, c2 //8, k=3, s=1, g=groups))
#         self.scconv2 = SCConv(
#             inplanes=c2 // 8, planes=c2 // 8, k=3, groups=groups, pooling_r=4)
#         self.k1_3 = nn.Sequential(
#             Conv(c2 // 8, c2 // 8, k=3, s=1, g=groups))
#         self.scconv3 = SCConv(
#             inplanes=c2 // 8, planes=c2 // 8, k=5, groups=groups, pooling_r=4)
#         self.endconv=Conv(c2,c2,k=1,s=1)
#     def forward(self, x):
#         x=self.conv1(x)
#         b, c, h, w = x.data.size()
#         x1,x4,x2,x3=torch.split(x, c // 4, dim=1)
#         x1=torch.cat([x1,x4], dim=1)
#         # x2=self.conv2(x1)
#         # x3=self.conv3(x1)
#
#         b1, c1, h1, w1 = x1.data.size()
#         x1_1, x1_2 = torch.split(x1, c1 // 2, dim=1)
#         out1_a= self.k1_1(x1_1)
#         out1_b = self.scconv1(x1_2)
#         out1 = torch.cat([out1_a, out1_b], dim=1)
#
#         b2, c2, h2, w2 = x2.data.size()
#         x2_1, x2_2 = torch.split(x2, c2 // 2, dim=1)
#         out2_a = self.k1_2(x2_1)
#         out2_b = self.scconv2(x2_2)
#         out2 = torch.cat([out2_a, out2_b], dim=1)
#
#         b3, c3, h3, w3 = x3.data.size()
#         x3_1, x3_2 = torch.split(x3, c3 // 2, dim=1)
#         out3_a = self.k1_3(x3_1)
#         out3_b = self.scconv3(x3_2)
#         out3 = torch.cat([out3_a, out3_b], dim=1)
#
#         out =torch.cat([out1, out2,out3], dim=1)
#         # #out=self.endconv(out)
#         # # shuffle
#         # #
#         # b, n, h, w = out.data.size()
#         # b_n = b * n // 2
#         # y = out.reshape(b_n, 2, h * w)
#         # y = y.permute(1, 0, 2)
#         # y = y.reshape(2, -1, n // 2, h, w)
#         # out=torch.cat((y[0], y[1]), 1)
#         return out

#channel258
# class GSConv(nn.Module):
#     """SCNet SCBottleneck
#     """
#
#     def __init__(self, c1, c2, stride=1,dilation=1,groups=1):
#         super().__init__()
#         # self.conv1= nn.Sequential(
#         #     Conv(c1, c2//2, k=1, s=1, g=groups)
#         # )
#         self.conv1 = nn.Sequential(
#             Conv(c1, c2//3, k=1, s=1, g=groups)
#         )
#         self.conv2 = nn.Sequential(
#             Conv(c2//3, c2 //3, k=3, s=1, g=groups)
#         )
#         self.conv3 = nn.Sequential(
#             Conv(c2//3, c2 // 3, k=5, s=1, g=groups)
#         )
#         self.k1_1= nn.Sequential(
#             Conv(c2//6, c2//6, k=1, s=1, g=groups))
#         self.scconv1 = SCConv(
#             inplanes=c2//6, planes=c2//6,k=1,groups=groups,pooling_r=4)
#         self.k1_2  = nn.Sequential(
#             Conv(c2//6, c2 //6, k=3, s=1, g=groups))
#         self.scconv2 = SCConv(
#             inplanes=c2 // 6, planes=c2 // 6, k=3, groups=groups, pooling_r=4)
#         self.k1_3 = nn.Sequential(
#             Conv(c2 // 6, c2 // 6, k=5, s=1, g=groups))
#         self.scconv3 = SCConv(
#             inplanes=c2 // 6, planes=c2 // 6, k=5, groups=groups, pooling_r=4)
#         self.endconv=Conv(c2,c2,k=1,s=1)
#     def forward(self, x):
#         x1=self.conv1(x)
#         # b, c, h, w = x.data.size()
#         # x1,x2,x3=torch.split(x, c // 3, dim=1)
#         x2=self.conv2(x1)
#         x3=self.conv3(x1)
#
#         b1, c1, h1, w1 = x1.data.size()
#         x1_1, x1_2 = torch.split(x1, c1 // 2, dim=1)
#         out1_a= self.k1_1(x1_1)
#         out1_b = self.scconv1(x1_2)
#         out1 = torch.cat([out1_a, out1_b], dim=1)
#
#         b2, c2, h2, w2 = x2.data.size()
#         x2_1, x2_2 = torch.split(x2, c2 // 2, dim=1)
#         out2_a = self.k1_2(x2_1)
#         out2_b = self.scconv2(x2_2)
#         out2 = torch.cat([out2_a, out2_b], dim=1)
#
#         b3, c3, h3, w3 = x3.data.size()
#         x3_1, x3_2 = torch.split(x3, c3 // 2, dim=1)
#         out3_a = self.k1_3(x3_1)
#         out3_b = self.scconv3(x3_2)
#         out3 = torch.cat([out3_a, out3_b], dim=1)
#
#         out =torch.cat([out1, out2,out3], dim=1)
#        # out=self.endconv(out)
#         # # shuffle
#         # #
#         # b, n, h, w = out.data.size()
#         # b_n = b * n // 3
#         # y = out.reshape(b_n, 3, h * w)
#         # y = y.permute(1, 0, 2)
#         # y = y.reshape(3, -1, n // 3, h, w)
#         # out = torch.cat((y[0], y[1], y[2]), 1)
#         return out
class GSConv(nn.Module):
    """SCNet SCBottleneck
    """

    def __init__(self, c1, c2, stride=1,dilation=1,groups=1):
        super().__init__()
        # self.conv1= nn.Sequential(
        #     Conv(c1, c2//2, k=1, s=1, g=groups)
        # )
        self.conv1 = nn.Sequential(
            Conv(c1, c2//2, k=1, s=1, g=groups)
        )
        self.conv2 = nn.Sequential(
            Conv(c1, c2 //2, k=3, s=1, g=groups)
        )
        self.k1_1= nn.Sequential(
            Conv(c2//4, c2//4, k=1, s=1, g=groups))
        self.scconv1 = SCConv(
            inplanes=c2//4, planes=c2//4,k=1,groups=groups,pooling_r=4)
        self.k1_2  = nn.Sequential(
            Conv(c2//4, c2 //4, k=3, s=1, g=groups))
        self.scconv2 = SCConv(
            inplanes=c2 // 4, planes=c2 // 4, k=3, groups=groups, pooling_r=4)
        self.endconv=Conv(c2,c2,k=1,s=1)
    def forward(self, x):
        x1=self.conv1(x)
        # b, c, h, w = x.data.size()
        # x1,x2,x3=torch.split(x, c // 3, dim=1)
        x2=self.conv2(x)
        b1, c1, h1, w1 = x1.data.size()
        x1_1,x1_2= torch.split(x1, c1 // 2, dim=1)
        out1_a= self.k1_1(x1_1)
        out1_b = self.scconv1(x1_2)
        out1 = torch.cat([out1_a, out1_b], dim=1)

        b2, c2, h2, w2 = x2.data.size()
        x2_1,x2_2 = torch.split(x2, c2 // 2, dim=1)
        out2_a = self.k1_2(x2_1)
        out2_b = self.scconv2(x2_2)
        out2 = torch.cat([out2_a, out2_b], dim=1)

        out =torch.cat([out1, out2], dim=1)
       # out=self.endconv(out)
        # # shuffle
        # #
        # b, n, h, w = out.data.size()
        # b_n = b * n // 3
        # y = out.reshape(b_n, 3, h * w)
        # y = y.permute(1, 0, 2)
        # y = y.reshape(3, -1, n // 3, h, w)
        # out = torch.cat((y[0], y[1], y[2]), 1)
        return out


# class GSConv(nn.Module):
#     """SCNet SCBottleneck
#     """
#
#     def __init__(self, c1, c2, stride=1,dilation=1,groups=1):
#         super().__init__()
#         self.conv1= nn.Sequential(
#             Conv(c1, c2, k=1, s=1, g=groups)
#         )
#         self.conv2 = nn.Sequential(
#             Conv(c2, c2, k=3, s=1, g=groups)
#         )
#         self.conv3 = nn.Sequential(
#             Conv(c2, c2, k=5, s=1, g=groups)
#         )
#         self.k1_1= nn.Sequential(
#             Conv(c2//2, c2//2, k=1, s=1, g=groups))
#         self.scconv1 = SCConv(
#             inplanes=c2//2, planes=c2//2,k=1,groups=groups,pooling_r=4)
#         self.k1_2  = nn.Sequential(
#             Conv(c2//2, c2 //2, k=3, s=1, g=groups))
#         self.scconv2 = SCConv(
#             inplanes=c2 // 2, planes=c2 // 2, k=3, groups=groups, pooling_r=4)
#         self.k1_3 = nn.Sequential(
#             Conv(c2 // 2, c2 // 2, k=3, s=1, g=groups))
#         self.scconv3 = SCConv(
#             inplanes=c2 // 2, planes=c2 // 2, k=5, groups=groups, pooling_r=4)
#
#         self.convend=nn.Sequential(
#             Conv(c2*3, c2, k=1, s=1, g=groups)
#         )
#
#     def forward(self, x):
#         x1=self.conv1(x)
#         x2=self.conv2(x1)
#         x3=self.conv3(x1)
#
#         b1, c1, h1, w1 = x1.data.size()
#         x1_1, x1_2 = torch.split(x1, c1 // 2, dim=1)
#         out1_a= self.k1_1(x1_1)
#         out1_b = self.scconv1(x1_2)
#         out1 = torch.cat([out1_a, out1_b], dim=1)
#
#         b2, c2, h2, w2 = x2.data.size()
#         x2_1, x2_2 = torch.split(x2, c2 // 2, dim=1)
#         out2_a = self.k1_2(x2_1)
#         out2_b = self.scconv2(x2_2)
#         out2 = torch.cat([out2_a, out2_b], dim=1)
#
#         b3, c3, h3, w3 = x3.data.size()
#         x3_1, x3_2 = torch.split(x3, c3 // 2, dim=1)
#         out3_a = self.k1_3(x3_1)
#         out3_b = self.scconv3(x3_2)
#         out3 = torch.cat([out3_a, out3_b], dim=1)
#
#         out =torch.cat([out1, out2,out3], dim=1)
#         out=self.convend(out)
#
#         return out




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

class Ty(nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        self.factor = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                               nn.Flatten(),
                               nn.Linear(out_channels, 1))
    def forward(self,x):
        return self.factor(x)


if __name__ == '__main__':
    x = torch.randn(50, 512, 7, 7)
    # y=torch.randn(50, 512, 7, 7)
    # S=CoordAtt(in_channels=512,out_channels=512)
    #S = GSConv(512,256)
    # S=Ty(out_channels=512)
    # t = torch.sigmoid(S(x)).view(-1, 1, 1, 1)
    # print(t.size())
    # z=x*t+y
    # print(z.size())
    # anchors=torch.randn(4,4)
    # gt_bboxes=torch.randn(2,4)
    # candidate_idxs=torch.randn(2,4)
    # num_gt, num_bboxes = gt_bboxes.size(0), anchors.size(0)
    # anchors_cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
    # anchors_cy = (anchors[:, 1] + anchors[:, 3]) / 2.0
    # for gt_idx in range(num_gt):
    #     candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
    # ep_anchors_cx = anchors_cx.view(1, -1).expand(
    #     num_gt, num_bboxes).contiguous().view(-1)
    # ep_anchors_cy = anchors_cy.view(1, -1).expand(
    #     num_gt, num_bboxes).contiguous().view(-1)
    # candidate_idxs = candidate_idxs.view(-1)
    #
    # # calculate the left, top, right, bottom distance between positive
    # # bbox center and gt side
    # l_ = ep_anchors_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
    # t_ = ep_anchors_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
    # r_ = gt_bboxes[:, 2] - ep_anchors_cx[candidate_idxs].view(-1, num_gt)
    # b_ = gt_bboxes[:, 3] - ep_anchors_cy[candidate_idxs].view(-1, num_gt)
    # is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01

import torch

# a = torch.tensor([True, False, True])
# b = torch.tensor([False, True, True])
#
# result = ~(a & ~b)
a=torch.tensor([[0.2,0.3,0.4],
               [0.1,0.1,0.5]])
print(a.max(dim=1))