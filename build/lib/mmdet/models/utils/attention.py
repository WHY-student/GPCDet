import torch
import torch.nn as nn
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


class Channel_Att(nn.Module):
    def __init__(self, channels):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        #improve1 不sig直接输出
        #return x
        #improve2 sig后直接输出
        #x = torch.sigmoid(x)
        return torch.sigmoid(x)*residual

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
        return out*x

class ChannelSpatial(nn.Module):
    def __init__(self, channels):
        super(ChannelSpatial,self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.act = nn.Sigmoid()
    def forward(self, x):
        res=x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        weight_bn = weight_bn.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        weight_bn=weight_bn.expand_as(x)

        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.act(self.conv2d(out))
        out=out.expand_as(x)
        weight = weight_bn+out
        #return out
        return torch.sigmoid(weight*x)*res


class NAMAttention(nn.Module):
    def __init__(self, channels,device=1,chann=True,spatial=True):
        super(NAMAttention, self).__init__()
        #self.cp=ChannelSpatial(channels)
        self.spatial=SpatialAttentionModule()
        self.Channel_Att = Channel_Att(channels)
        #self.bn = nn.BatchNorm2d(channels, affine=True)
        #self.conv=nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        #self.gate = nn.Sigmoid()
        # self.Spatial_Att=Spatial_Att(channels,device)
        self.w = nn.Parameter(torch.tensor([1.0]),requires_grad=True)  # 可训练权重
    def forward(self, x):
        #return self.cp(x)
        #x2 = self.spatial(x)
        x1=self.Channel_Att(x)
        return x1
       # return x2
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


class CoordAtt(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()
        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        weight_bn = self.bn1.weight.data.abs() / torch.sum(self.bn1.weight.data.abs())
        out=out.permute(0,2,3,1).contiguous()
        out=torch.mul(weight_bn, out)
        out=out.permute(0, 3, 1, 2).contiguous()
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = self.conv2(x_h)
        out_w = self.conv3(x_w)
        w=torch.sigmoid_(out_h+out_w)
        return x*w

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
#         x_h, x_w = torch.split(out, [H, W], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#         out_h = self.conv2(x_h)
#         out_w = self.conv3(x_w)
#         w=torch.sigmoid_(out_h+out_w)
#         return x*w
#
