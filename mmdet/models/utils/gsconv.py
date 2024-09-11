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
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    # def forward(self, x):
    #     return self.act(self.bn(self.conv(x)))
    #
    # def forward_fuse(self, x):
    #     return self.act(self.conv(x))


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
class GSConv2(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1,g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.conv = Conv(c1, c_, k, s, None, g, act)
        self.cv1 = Conv(c_, c_//2, 5, 1, None, c_//2, act)
        self.cv2 = Conv(c_, c_//2, 1, 1, None, c_//2, act)

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
#     # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=1, s=1,g=1, act=True):
#         super().__init__()
#         c_ = c2 // 3
#         self.conv = Conv(c1, c_, k, s, None, g, act)
#         self.cv1 = Conv(c_, c_, 5, 1, None, c_, act)
#         self.cv2 = Conv(c_, c_, 3, 1, None, c_, act)
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
#         b_n = b * n // 3
#         y = x3.reshape(b_n, 3, h * w)
#         y = y.permute(1, 0, 2)
#         y = y.reshape(3, -1, n // 3, h, w)
#         y=torch.cat((y[0], y[1],y[2]), 1)
#         return y
# #
# class SCConv(nn.Module):
#     def __init__(self, inplanes, planes,k=3,groups=1, pooling_r=4):
#         super().__init__()
#         self.k2 = nn.Sequential(
#                     nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
#                     Conv(inplanes, planes, k=k, s=1,g=groups),
#                     )
#         self.k3 = nn.Sequential(
#                     Conv(inplanes, planes, k=k, s=1,g=groups),
#                   #  norm_layer(planes),
#                     )
#         self.k4 = nn.Sequential(
#             Conv(inplanes, planes, k=k, s=1, g=groups),
#                  #   norm_layer(planes),
#                     )
#
#     def forward(self, x):
#         identity = x
#
#         out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
#         out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
#         out = self.k4(out) # k4
#         return out

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
#             Conv(c1, c2//2, k=1, s=1, g=groups)
#         )
#         self.conv2 = nn.Sequential(
#             Conv(c1, c2 //2, k=3, s=1, g=groups)
#         )
#         self.k1_1= nn.Sequential(
#             Conv(c2//4, c2//4, k=1, s=1, g=groups))
#         self.scconv1 = SCConv(
#             inplanes=c2//4, planes=c2//4,k=1,groups=groups,pooling_r=4)
#         self.k1_2  = nn.Sequential(
#             Conv(c2//4, c2 //4, k=3, s=1, g=groups))
#         self.scconv2 = SCConv(
#             inplanes=c2 // 4, planes=c2 // 4, k=3, groups=groups, pooling_r=4)
#         self.endconv=Conv(c2,c2,k=1,s=1)
#     def forward(self, x):
#         x1=self.conv1(x)
#         # b, c, h, w = x.data.size()
#         # x1,x2=torch.split(x, c // 2, dim=1)
#         x2=self.conv2(x)
#         b1, c1, h1, w1 = x1.data.size()
#         x1_1,x1_2= torch.split(x1, c1 // 2, dim=1)
#         out1_a= self.k1_1(x1_1)
#         out1_b = self.scconv1(x1_2)
#         out1 = torch.cat([out1_a, out1_b], dim=1)
#
#         b2, c2, h2, w2 = x2.data.size()
#         x2_1,x2_2 = torch.split(x2, c2 // 2, dim=1)
#         out2_a = self.k1_2(x2_1)
#         out2_b = self.scconv2(x2_2)
#         out2 = torch.cat([out2_a, out2_b], dim=1)
#
#         out =torch.cat([out1, out2], dim=1)
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


# class GSConv(nn.Module):
#     # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=1, s=1,g=1, act=True):
#         super().__init__()
#         self.l_conv = ConvModule(
#             c1,
#             c2,
#             1,
#             inplace=False)
#         self.conv1=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=1)
#         self.conv2=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5,stride=1,padding=2)
#
#     #improve1
#     def forward(self, x):
#        x=self.l_conv(x)
#        n,c,h,w=x.data.size()
#        x1, x2 = torch.split(x, c // 2, dim=1)
#
#        x1_1,x1_2=torch.split(x1,c//4,dim=1)
#
#        # 沿着第二个维度求平均值，即在c维度上进行平均池化
#        mean_x1 = torch.mean(x1_1, dim=1)
#        # 将结果的维度由(n, w, h)变为(n, 1, w, h)
#        mean_x1 = torch.unsqueeze(mean_x1, 1)
#
#        # 沿着第二个维度求平均值，即在c维度上进行平均池化
#        mean_x2 = torch.mean(x1_2, dim=1)
#        # 将结果的维度由(n, w, h)变为(n, 1, w, h)
#        mean_x2 = torch.unsqueeze(mean_x2, 1)
#
#        weight1=self.conv1(mean_x1).sigmoid()
#        weight2=self.conv2(mean_x2).sigmoid()
#
#        x1_1=x1_1*weight1
#        x1_2=x1_2*weight2
#        x1=torch.cat((x1_1,x1_2),1)
#        #shuffle1
#        b1,n1,h1,w1=x1.data.size()
#        b1_n=b1*n1//2
#        y=x1.reshape(b1_n,2,h*w)
#        y=y.permute(1,0,2)
#        y=y.reshape(2,-1,n1//2,h,w)
#        x1=torch.cat((y[0],y[1]),1)
#
#        x=torch.cat((x1,x2),1)
#        # shuffle2
#        b2, n2, h2, w2 = x.data.size()
#        b2_n = b2 * n2 // 2
#        z = x.reshape(b2_n, 2, h2 * w2)
#        z = z.permute(1, 0, 2)
#        z = z.reshape(2, -1, n2 // 2, h2, w2)
#        x = torch.cat((z[0],z[1]), 1)
#        return x

# class GSConv(nn.Module):
#     # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=1, s=1,g=1, act=True):
#         super().__init__()
#         self.l_conv = ConvModule(
#             c1,
#             c2,
#             1,
#             inplace=False)
#         self.conv3=Conv(c2//4, c2//4, 3, 1, None, c2//4, act)
#         self.conv5=Conv(c2//4, c2//4, 5, 1, None, c2//4, act)
#
#     #improve1
#     def forward(self, x):
#        x=self.l_conv(x)
#        n,c,h,w=x.data.size()
#        x1, x2 = torch.split(x, c // 2, dim=1)
#
#        x1_1,x1_2=torch.split(x1,c//4,dim=1)
#
#        x1_1=self.conv3(x1_1)
#        x1_2=self.conv5(x1_2)
#        x1=torch.cat((x1_1,x1_2),1)
#
#        #shuffle1
#        b1,n1,h1,w1=x1.data.size()
#        b1_n=b1*n1//2
#        y=x1.reshape(b1_n,2,h*w)
#        y=y.permute(1,0,2)
#        y=y.reshape(2,-1,n1//2,h,w)
#        x1=torch.cat((y[0],y[1]),1)
#
#        x=torch.cat((x1,x2),1)
#        # shuffle2
#        b2, n2, h2, w2 = x.data.size()
#        b2_n = b2 * n2 // 2
#        z = x.reshape(b2_n, 2, h2 * w2)
#        z = z.permute(1, 0, 2)
#        z = z.reshape(2, -1, n2 // 2, h2, w2)
#        x = torch.cat((z[0],z[1]), 1)
#        return x


class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1,g=1, act=True):
        super().__init__()
        self.l_conv = ConvModule(
            c1,
            c2//2,
            1,
            inplace=False)
        self.conv3 = Conv(c2 // 2, c2 // 2, 3, 1, None, c2 // 2, act)
        self.conv5 = Conv(c2 // 2, c2 // 2, 5, 1, None, c2 // 2, act)

    #improve1
    def forward(self, x):
       x=self.l_conv(x)
       n,c,h,w=x.data.size()

       x1_gc=self.conv3(x)
       x2_gc=self.conv5(x)

       x1=torch.cat((x,x1_gc),dim=1)

       #shuffle1
       b1,n1,h1,w1=x1.data.size()
       b1_n=b1*n1//2
       y=x1.reshape(b1_n,2,h*w)
       y=y.permute(1,0,2)
       y=y.reshape(2,-1,n1//2,h,w)
       x1=torch.cat((y[0],y[1]),1)

       x2=torch.cat((x,x2_gc),dim=1)
       # shuffle2
       b2, n2, h2, w2 = x2.data.size()
       b2_n = b2 * n2 // 2
       z = x2.reshape(b2_n, 2, h2 * w2)
       z = z.permute(1, 0, 2)
       z = z.reshape(2, -1, n2 // 2, h2, w2)
       x2 = torch.cat((z[0],z[1]), 1)

       x=x1+x2
       return x

