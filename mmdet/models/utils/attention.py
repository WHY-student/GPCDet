import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn import Softmax
from collections import OrderedDict
from mmcv.cnn import normal_init
from mmcv.runner import BaseModule
import torch.nn.functional as F
import math



# class Spatial_Att(nn.Module):
#     def __init__(self,channel,device=1):
#         super(Spatial_Att, self).__init__()
#         self.device=device
#
#         # self.gn = nn.GroupNorm(channel, channel)
#         # self.sweight = Parameter(torch.zeros(1, channel, 1, 1))
#         # self.sbias = Parameter(torch.ones(1, channel, 1, 1))
#         # self.sigmoid = nn.Sigmoid()
#         #self.device="cuda:"+str(device)
#     def forward(self,x):
#         # spatial attention
#         # x_spatial = self.gn(x)  # bs*G,c//(2*G),h,w
#         # x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
#         # x_spatial = x * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w
#         b,c,h,w=x.shape
#         bn3 = nn.BatchNorm2d(h*w, affine=True).to(torch.device(self.device))
#         residual=x
#         x=x.permute(0,2,3,1).reshape(b,h*w,1,c).contiguous()
#         x=bn3(x)
#
#         weight_bn=bn3.weight.data.abs()/torch.sum(bn3.weight.data.abs())
#
#         x=x.permute(0,3,2,1).contiguous()
#         x=torch.mul(weight_bn,x)
#         x=x.reshape(b,c,h,w).contiguous()
#         x=torch.sigmoid(x)*residual
#         return x

#
# class Channel_Spatial(nn.Module):
#     def __init__(self, channels, device=1):
#         super(Channel_Spatial, self).__init__()
#         self.channels = channels
#         self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
#         self.device=device
#     def forward(self, x):
#         residual = x
#
#         #channel
#         x1 = self.bn2(x)
#         weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
#         x1 = x1.permute(0, 2, 3, 1).contiguous()
#         x1 = torch.mul(weight_bn, x1)
#         x1 = x1.permute(0, 3, 1, 2).contiguous()
#
#         #spatial
#         b, c, h, w = x.shape
#         bn3 = nn.BatchNorm2d(h * w, affine=True).to(torch.device(self.device))
#         x2 = x.permute(0, 2, 3, 1).reshape(b, h * w, 1, c).contiguous()
#         x2 = bn3(x2)
#         weight_bn = bn3.weight.data.abs() / torch.sum(bn3.weight.data.abs())
#         x2 = x2.permute(0, 3, 2, 1).contiguous()
#         x2 = torch.mul(weight_bn, x2)
#         x2 = x2.reshape(b, c, h, w).contiguous()
#
#         #improve1
#         #x = torch.sigmoid(x1+x2) * residual
#         #improve2
#         #x=(x1+x2)*residual
#         return x

# #improve2
# class Channel_Spatial(nn.Module):
#     def __init__(self, channels, device=1):
#         super(Channel_Spatial, self).__init__()
#         self.channels = channels
#         self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
#         self.device=device
#         self.w=nn.Parameter(torch.tensor([0.01]),requires_grad=True)  # 可训练权重
#     def forward(self, x):
#         residual = x
#
#         #channel
#         x1 = self.bn2(x)
#         weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
#         x1 = x1.permute(0, 2, 3, 1).contiguous()
#         x1 = torch.mul(weight_bn, x1)
#         x1 = x1.permute(0, 3, 1, 2).contiguous()
#
#         #spatial
#         b, c, h, w = x.shape
#         bn3 = nn.BatchNorm2d(h * w, affine=True).to(torch.device(self.device))
#         x2 = x.permute(0, 2, 3, 1).reshape(b, h * w, 1, c).contiguous()
#         x2 = bn3(x2)
#         weight_bn = bn3.weight.data.abs() / torch.sum(bn3.weight.data.abs())
#         x2 = x2.permute(0, 3, 2, 1).contiguous()
#         x2 = torch.mul(weight_bn, x2)
#         x2 = x2.reshape(b, c, h, w).contiguous()
#
#         # x = torch.sigmoid((x1+x2)/2)*residual   #improve2
#         weight=torch.sigmoid(self.w[0])
#         x = torch.sigmoid(x1*weight+x2*(1-weight))*residual #improve3
#         return x

# class ChannelAttentionModule(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(ChannelAttentionModule, self).__init__()
#         mid_channel = channel // reduction
#         # 使用自适应池化缩减map的大小，保持通道不变
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.shared_MLP = nn.Sequential(
#             nn.Linear(in_features=channel, out_features=mid_channel),
#             nn.ReLU(),
#             nn.Linear(in_features=mid_channel, out_features=channel)
#         )
#         self.sigmoid = nn.Sigmoid()
#         # self.act=SiLU()
#
#     def forward(self, x):
#         avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
#         maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
#         return self.sigmoid(avgout + maxout)

# 4. NAMAttention https://blog.csdn.net/qq_38668236/article/details/126503665

#
# class Channel_Att(nn.Module):
#     def __init__(self, channels):
#         super(Channel_Att, self).__init__()
#         self.channels = channels
#         self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
#     def forward(self, x):
#         residual = x
#         x = self.bn2(x)
#         weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
#         x = x.permute(0, 2, 3, 1).contiguous()
#         x = torch.mul(weight_bn, x)
#         x = x.permute(0, 3, 1, 2).contiguous()
#         #improve1 不sig直接输出
#         #return x
#         #improve2 sig后直接输出
#         #x = torch.sigmoid(x)
#         return torch.sigmoid(x)*residual
#
# class SpatialAttentionModule(nn.Module):
#     def __init__(self):
#         super(SpatialAttentionModule, self).__init__()
#         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
#         self.act = nn.Sigmoid()
#     def forward(self, x):
#         avgout = torch.mean(x, dim=1, keepdim=True)
#         maxout, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.act(self.conv2d(out))
#         return out*x
#
# class ChannelSpatial(nn.Module):
#     def __init__(self, channels):
#         super(ChannelSpatial, self).__init__()
#         self.channels = channels
#         self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
#         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, dilation=3,stride=1, padding=3)
#         #self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         res = x
#         x = self.bn2(x)
#         y = x
#         weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
#         x = x.permute(0, 2, 3, 1).contiguous()
#         x = torch.mul(weight_bn, x)
#         x = x.permute(0, 3, 1, 2).contiguous()
#         avgout = torch.mean(y, dim=1, keepdim=True)
#         maxout, _ = torch.max(y, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.conv2d(out)
#         weight = self.act(x * out)
#         # weight2=self.act(out)
#         return weight * res
# class ChannelSpatial(nn.Module):
#     def __init__(self, channels):
#         super(ChannelSpatial,self).__init__()
#         self.channels = channels
#         self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
#         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, dilation=1,stride=1, padding=3)
#         self.act = nn.Sigmoid()
#     def forward(self, x):
#         res=x
#         x = self.bn2(x)
#         weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
#         weight_bn=weight_bn.unsqueeze(0).unsqueeze(2).unsqueeze(3)
#         avgout = torch.mean(x, dim=1, keepdim=True)
#         maxout, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.act(self.conv2d(out))
#         weight=self.act(torch.mul(weight_bn, out))
#         #return out
#         return weight*x

# class NAMAttention(nn.Module):
#     def __init__(self, channels,device=1,chann=True,spatial=True):
#         super(NAMAttention, self).__init__()
#         self.cp=ChannelSpatial(channels)
#         self.spatial=SpatialAttentionModule()
#         self.Channel_Att = Channel_Att(channels)
#         #self.bn = nn.BatchNorm2d(channels, affine=True)
#         #self.conv=nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
#         #self.gate = nn.Sigmoid()
#         # self.Spatial_Att=Spatial_Att(channels,device)
#         self.w = nn.Parameter(torch.tensor([1.0]),requires_grad=True)  # 可训练权重
#     def forward(self, x):
#         return self.cp(x)
        #x2 = self.spatial(x)
        #x1=self.Channel_Att(x)
        #return x1
        #return x2
        #w=torch.sigmoid(self.w[0])
        #return w*x1+(1-w)*x2
        # weight=torch.sigmoid(self.w[0]);
        # weight1=torch.sigmoid(self.w[1]);
        # out=2*(x1*weight+x2*weight1)
        #weigh=torch.exp(self.w)/torch.sum(torch.exp(self.w))

        #return x1*self.w[0]+x2*self.w[1]
        #return x1*weigh[0]+x2*weigh[1]+x*weigh[2]
        # x1 = self.Channel_Att(x)
        # #x2 = self.Spatial_Att(x1*w[0]+res*w[1]) 效果很棒
        # weight=torch.sigmoid(self.w[0])
        #
        # #x2=self.Spatial_Att(x1*weight+(1-weight)*x)
        # #x2 = self.Spatial_Att(x1 * weight + weight * x)
        # x2=self.Spatial_Att(x)
        # #out=torch.sigmoid(self.w[0])*x1+(1-torch.sigmoid(self.w[0]))*x2
        # #weight2=torch.sigmoid(self.w2[0])
        # #x2=x2*weight+x*(1-weight)
        # return self.bn(x1*weight+x2*weight)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


# class CoordAtt(nn.Module):
#
#     def __init__(self, in_channels, out_channels, reduction=32):
#         super(CoordAtt, self).__init__()
#         self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
#         temp_c = max(8, in_channels // reduction)
#         self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)
#
#         self.bn1 = nn.BatchNorm2d(temp_c)
#         self.act1 = h_swish()
#         self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
#         self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         n, c, H, W = x.shape
#         x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
#         x_cat = torch.cat([x_h, x_w], dim=2)
#         out = self.act1(self.bn1(self.conv1(x_cat)))
#         weight_bn = self.bn1.weight.data.abs() / torch.sum(self.bn1.weight.data.abs())
#         out=out.permute(0,2,3,1).contiguous()
#         out=torch.mul(weight_bn, out)
#         out=out.permute(0, 3, 1, 2).contiguous()
#         x_h, x_w = torch.split(out, [H, W], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#         out_h = self.conv2(x_h)
#         out_w = self.conv3(x_w)
#         w=torch.sigmoid_(out_h+out_w)
#         return x*w

# #原版CA
# class CoordAtt(nn.Module):
#     def __init__(self, in_channels, out_channels, groups=32):
#         super(CoordAtt, self).__init__()
#
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#
#         mip = max(8, out_channels // groups)
#
#         self.conv1 = nn.Conv2d(out_channels, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.conv2 = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
#         self.conv3 = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
#         self.relu = h_swish()
#
#     def forward(self, x):
#         x = self.conv(x)
#         identity = x
#         n,c,h,w = x.size()
#         x_h = self.pool_h(x)
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)
#
#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.relu(y)
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#
#         x_h = self.conv2(x_h).sigmoid()
#         x_w = self.conv3(x_w).sigmoid()
#         x_h = x_h.expand(-1, -1, h, w)
#         x_w = x_w.expand(-1, -1, h, w)
#
#         y = identity * x_w * x_h
#
#         return y

class CoordAtt_Backbone(nn.Module):

    def __init__(self,out_channels):
        super(CoordAtt_Backbone, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        self.conv0=nn.Conv2d(out_channels,1,kernel_size=1,stride=1,padding=0)
        self.conv1 = ConvModule(
            1,
            1,
            1,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None,
            inplace=False)
        self.act1 = h_swish()
        self.conv2 = ConvModule(
            1,
            1,
            1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None,
            inplace=False)
        self.conv3 = ConvModule(
            1,
            1,
            1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None,
            inplace=False)


    def forward(self, x):
        n, c, H, W = x.shape
        feature=self.conv0(x)
        x_h, x_w = self.pool_h(feature), self.pool_w(feature).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.conv1(x_cat))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = self.conv2(x_h).sigmoid()
        out_w = self.conv3(x_w).sigmoid()
        return x*out_w*out_h

# 不聚和
# class CoordAtt(nn.Module):
#
#     def __init__(self, in_channels, out_channels):
#         super(CoordAtt, self).__init__()
#         self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
#         self.conv=ConvModule(
#             in_channels,
#             out_channels,
#             1,
#             conv_cfg=None,
#             norm_cfg=dict(type='BN', requires_grad=True),
#             act_cfg=None,
#             inplace=False)
#         self.conv0=nn.Conv2d(out_channels,1,kernel_size=1,stride=1,padding=0)
#         self.conv1 = ConvModule(
#             1,
#             1,
#             1,
#             conv_cfg=None,
#             norm_cfg=dict(type='BN', requires_grad=True),
#             act_cfg=None,
#             inplace=False)
#         self.act1 = h_swish()
#         self.conv2 = ConvModule(
#             1,
#             1,
#             1,
#             conv_cfg=None,
#             norm_cfg=None,
#             act_cfg=None,
#             inplace=False)
#         self.conv3 = ConvModule(
#             1,
#             1,
#             1,
#             conv_cfg=None,
#             norm_cfg=None,
#             act_cfg=None,
#             inplace=False)
#
#
#     def forward(self, x):
#         x=self.conv(x)
#         n, c, H, W = x.shape
#         feature=self.conv0(x)
#         x_h, x_w = self.pool_h(feature), self.pool_w(feature)
#         # x_cat = torch.cat([x_h, x_w], dim=2)
#         # out = self.act1(self.conv1(x_cat))
#         # x_h, x_w = torch.split(out, [H, W], dim=2)
#         # x_w = x_w.permute(0, 1, 3, 2)
#         #out_h = self.conv2(x_h).sigmoid()
#         out_w = self.conv3(x_w).sigmoid()
#         return x*out_w


# #我的
class CoordAtt(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CoordAtt, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        self.conv=ConvModule(
            in_channels,
            out_channels,
            1,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None,
            inplace=False)
        self.conv0=nn.Conv2d(out_channels,1,kernel_size=1,stride=1,padding=0)
        self.conv1 = ConvModule(
            1,
            1,
            1,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None,
            inplace=False)
        self.act1 = h_swish()
        self.conv2 = ConvModule(
            1,
            1,
            1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None,
            inplace=False)
        self.conv3 = ConvModule(
            1,
            1,
            1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None,
            inplace=False)


    def forward(self, x):
        x=self.conv(x)
        n, c, H, W = x.shape
        feature=self.conv0(x)
        x_h, x_w = self.pool_h(feature), self.pool_w(feature).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.conv1(x_cat))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = self.conv2(x_h).sigmoid()
        out_w = self.conv3(x_w).sigmoid()
        return x*out_w*out_h

# 我的
class CoordAtt_forbfhead(nn.Module):

    def __init__(self, out_channels):
        super(CoordAtt_forbfhead, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        self.conv0 = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0)
        self.conv1 = ConvModule(
            1,
            1,
            1,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None,
            inplace=False)
        self.act1 = h_swish()
        self.conv2 = ConvModule(
            1,
            1,
            1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None,
            inplace=False)
        self.conv3 = ConvModule(
            1,
            1,
            1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None,
            inplace=False)

    def forward(self, x):
        n, c, H, W = x.shape
        feature = self.conv0(x)
        x_h, x_w = self.pool_h(feature), self.pool_w(feature).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.conv1(x_cat))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = self.conv2(x_h).sigmoid()
        out_w = self.conv3(x_w).sigmoid()
        return x * out_w * out_h

#改进版CA
# class CoordAtt(nn.Module):
#
#     def __init__(self, in_channels, out_channels):
#         super(CoordAtt, self).__init__()
#         self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
#         self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.conv0=nn.Conv2d(out_channels,1,kernel_size=1,stride=1,padding=0)
#         self.conv1 = nn.Conv2d(1,1, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(1)
#         self.act1 = h_swish()
#         self.conv2 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
#         self.conv3 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 normal_init(m, std=0.001)
#         normal_init(self.conv, std=0.01)
#         normal_init(self.conv0, std=0.01)
#         normal_init(self.conv1, std=0.01)
#         normal_init(self.conv2, std=0.01)
#         normal_init(self.conv3, std=0.01)
#         nn.init.ones_(self.bn1.weight)
#         nn.init.zeros_(self.bn1.bias)
#         nn.init.ones_(self.bn.weight)
#         nn.init.zeros_(self.bn.bias)
#
#     def forward(self, x):
#         x=self.conv(x)
#         x=self.bn(x)
#         n, c, H, W = x.shape
#         feature=self.conv0(x)
#         x_h, x_w = self.pool_h(feature), self.pool_w(feature).permute(0, 1, 3, 2)
#         x_cat = torch.cat([x_h, x_w], dim=2)
#         out = self.act1(self.bn1(self.conv1(x_cat)))
#         x_h, x_w = torch.split(out, [H, W], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#         out_h = self.conv2(x_h).sigmoid()
#         out_w = self.conv3(x_w).sigmoid()
#         return x*out_w*out_h


#并
# class CoordNAM(nn.Module):
#
#     def __init__(self, in_channels, out_channels):
#         super(CoordNAM, self).__init__()
#         self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
#         self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.conv0=nn.Conv2d(out_channels,1,kernel_size=1,stride=1,padding=0)
#         self.conv1 = nn.Conv2d(1,1, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(1)
#         self.act1 = h_swish()
#         self.conv2 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
#         self.conv3 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
#
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         # NAM
#         weight_bn = self.bn.weight.data.abs() / torch.sum(self.bn.weight.data.abs())
#         y = x.permute(0, 2, 3, 1).contiguous()
#         y = torch.mul(weight_bn, y)
#         y = y.permute(0, 3, 1, 2).contiguous()
#
#         # CA
#         n, c, H, W = x.shape
#         feature = self.conv0(x)
#         x_h, x_w = self.pool_h(feature), self.pool_w(feature).permute(0, 1, 3, 2)
#         x_cat = torch.cat([x_h, x_w], dim=2)
#         out = self.act1(self.bn1(self.conv1(x_cat)))
#         x_h, x_w = torch.split(out, [H, W], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#         out_h = self.conv2(x_h).sigmoid()
#         out_w = self.conv3(x_w).sigmoid()
#
#         return x * out_w * out_h * torch.sigmoid(y)

# # 串联
class CoordNAM(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CoordNAM, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv0=nn.Conv2d(out_channels,1,kernel_size=1,stride=1,padding=0)
        self.conv1 = nn.Conv2d(1,1, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(1)
        self.act1 = h_swish()
        self.conv2 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        # NAM
        weight_bn = self.bn.weight.data.abs() / torch.sum(self.bn.weight.data.abs())
        y = x.permute(0, 2, 3, 1).contiguous()
        y = torch.mul(weight_bn, y)
        y = y.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(y) * x
        # CA
        n, c, H, W = x.shape
        feature = self.conv0(x)
        x_h, x_w = self.pool_h(feature), self.pool_w(feature).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = self.conv2(x_h).sigmoid()
        out_w = self.conv3(x_w).sigmoid()
        x= x * out_w * out_h
        return x





class NAM(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(NAM, self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x=self.conv(x)
        x=self.bn(x)
        y=x
        #NAM
        weight_bn=self.bn.weight.data.abs()/torch.sum(self.bn.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return y*torch.sigmoid(x)


class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
        self.act = nn.Sigmoid()
        # self.act=nn.SiLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.act(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.act(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBAM, self).__init__()
        self.conv=ConvModule(
            in_channels,
            out_channels,
            1,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None,
            inplace=False)
        self.channel_attention = ChannelAttentionModule(out_channels)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        x=self.conv(x)
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class CBAMSPA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBAMSPA, self).__init__()
        self.conv=ConvModule(
            in_channels,
            out_channels,
            1,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None,
            inplace=False)
        self.channel_attention = ChannelAttentionModule(out_channels)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        x=self.conv(x)
        out = self.spatial_attention(x) * x
        return out



class GAMAttention(nn.Module):
    # https://paperswithcode.com/paper/global-attention-mechanism-retain-information
    def __init__(self, c1, c2, group=True, rate=4):
        super(GAMAttention, self).__init__()
        self.conv = ConvModule(
            c1,
            c2,
            1,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None,
            inplace=False)
        self.channel_attention = nn.Sequential(
            nn.Linear(c2, int(c2 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c2 / rate), c2)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c2, c2 // rate, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(c2, int(c2 / rate),
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(int(c2 / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2 // rate, c2, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(int(c2 / rate), c2,
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        x=self.conv(x)
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
        out = x * x_spatial_att
        return out


def channel_shuffle(x, groups=2):  ##shuffle channel
    # RESHAPE----->transpose------->Flatten
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out


class GAMSPAAttention(nn.Module):
    # https://paperswithcode.com/paper/global-attention-mechanism-retain-information
    def __init__(self, c1, c2, group=True, rate=4):
        super(GAMSPAAttention, self).__init__()
        self.conv = ConvModule(
            c1,
            c2,
            1,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None,
            inplace=False)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c2, c2 // rate, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(c2, int(c2 / rate),
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(int(c2 / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2 // rate, c2, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(int(c2 / rate), c2,
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        x = self.conv(x)
        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
        out = x * x_spatial_att
        return out


class ChannelGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel)
        )
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.avgpool(x).view(b, c)
        y = self.mlp(y)
        y = self.bn(y).view(b, c, 1, 1)
        return y.expand_as(x)


class SpatialGate(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=3, dilation_val=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel // reduction, channel // reduction, kernel_size, padding=dilation_val,
                      dilation=dilation_val),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel // reduction, kernel_size, padding=dilation_val,
                      dilation=dilation_val),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(channel // reduction, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.bn(y)
        return y.expand_as(x)


class BAM(nn.Module):
    def __init__(self, c1,channel):
        super(BAM, self).__init__()
        self.channel_attn = ChannelGate(channel)
        self.spatial_attn = SpatialGate(channel)
        self.conv = ConvModule(
            c1,
            channel,
            1,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None,
            inplace=False)

    def forward(self, x):
        x=self.conv(x)
        attn = F.sigmoid(self.channel_attn(x) + self.spatial_attn(x))
        return x + x * attn

class BAMSPA(nn.Module):
    def __init__(self, c1,channel):
        super(BAMSPA, self).__init__()
        self.conv=ConvModule(
            c1,
            channel,
            1,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None,
            inplace=False)
        self.spatial_attn = SpatialGate(channel)

    def forward(self, x):
        x=self.conv(x)
        attn = F.sigmoid(self.spatial_attn(x))
        return x + x * attn


class GCNet(nn.Module):
    def __init__(self, c1=256,inplanes=256, ratio=1/16, pooling_type="att", fusion_types=('channel_add')) -> None:
        super().__init__()

        assert pooling_type in ['avg', 'att']
        # assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_types

        self.conv = ConvModule(
            c1,
            inplanes,
            1,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None,
            inplace=False)

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None

        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':  # 这里其实就是空间注意力 最后得到一个b c 1 1的权重
            input_x = x
            input_x = input_x.view(batch, channel, height * width)  # -> b c h*w
            input_x = input_x.unsqueeze(1)  # -> b 1 c hw
            context_mask = self.conv_mask(x)  # b 1 h w
            context_mask = context_mask.view(batch, 1, height * width)  # b 1 hw
            context_mask = self.softmax(context_mask)  # b 1 hw
            context_mask = context_mask.unsqueeze(-1)  # b 1 hw 1
            context = torch.matmul(input_x, context_mask)  # b(1 c hw  *  1 hw 1) -> b 1 c 1
            context = context.view(batch, channel, 1, 1)  # b c 1 1
        else:
            context = self.avg_pool(x)  # b c 1 1
        return context

    def forward(self, x):
        x=self.conv(x)
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))  # 将权重进行放大缩小
            out = out * channel_mul_term  # 与x进行相乘
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out