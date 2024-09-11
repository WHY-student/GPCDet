# import mmcv
# import cv2 as cv
# import torch.nn
# from torch import nn
# from mmcv.cnn import ConvModule
#
# def autopad(k, p=None):  # kernel, padding
#     # Pad to 'same'
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p
#
# class Conv(nn.Module):
#     # Standard convolution
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU() if act else nn.Identity()
#
#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))
#
#     def forward_fuse(self, x):
#         return self.act(self.conv(x))
#     # def forward(self, x):
#     #     return self.act(self.bn(self.conv(x)))
#     #
#     # def forward_fuse(self, x):
#     #     return self.act(self.conv(x))
# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)
#
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
# # class CoordAtt(nn.Module):
# #
# #     def __init__(self, in_channels, out_channels, reduction=32):
# #         super(CoordAtt, self).__init__()
# #         self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
# #         self.bn1 = nn.BatchNorm2d(in_channels)
# #         self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
# #         self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
# #
# #     def forward(self, x):
# #         n, c, H, W = x.shape
# #         x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
# #         x_cat = torch.cat([x_h, x_w], dim=2)
# #         out = self.bn1(x_cat)
# #         x_h, x_w = torch.split(out, [H, W], dim=2)
# #         x_w = x_w.permute(0, 1, 3, 2)
# #         out_h = self.conv2(x_h)
# #         out_w = self.conv3(x_w)
# #         w=torch.sigmoid_(out_h+out_w)
# #         return x*w
#
# # class ChannelSpatial(nn.Module):
# #     def __init__(self, channels):
# #         super(ChannelSpatial,self).__init__()
# #         self.channels = channels
# #         self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
# #         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
# #         self.act = nn.Sigmoid()
# #     def forward(self, x):
# #         res=x
# #         x = self.bn2(x)
# #         weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
# #         weight_bn = weight_bn.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
# #         #weight_bn=weight_bn.expand_as(x)
# #
# #         avgout = torch.mean(x, dim=1, keepdim=True)
# #         maxout, _ = torch.max(x, dim=1, keepdim=True)
# #         out = torch.cat([avgout, maxout], dim=1)
# #         out = self.act(self.conv2d(out))
# #         #out=out.expand_as(x)
# #         weight = weight_bn*out
# #         #return out
# #         return torch.sigmoid(weight)*res
# # class ChannelSpatial(nn.Module):
# #     def __init__(self, channels):
# #         super(ChannelSpatial,self).__init__()
# #         self.channels = channels
# #         self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
# #         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=4, dilation=2,stride=1, padding=3)
# #         self.act = nn.Sigmoid()
# #     def forward(self, x):
# #         res=x
# #         x = self.bn2(x)
# #         weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
# #         avgout = torch.mean(x, dim=1, keepdim=True)
# #         maxout, _ = torch.max(x, dim=1, keepdim=True)
# #         out = torch.cat([avgout, maxout], dim=1)
# #         out = self.conv2d(out)
# #         weight = self.act(weight_bn*out)
# #         #return out
# #         return weight*x
# # class ChannelSpatial(nn.Module):
# #     def __init__(self, channels):
# #         super(ChannelSpatial,self).__init__()
# #         self.channels = channels
# #         self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
# #         #self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=4, dilation=2,stride=1, padding=3)
# #         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=4, dilation=2, stride=1, padding=2)
# #         self.act = nn.Sigmoid()
# #     def forward(self, x):
# #         res=x
# #         x = self.bn2(x)
# #         weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
# #         weight_bn=weight_bn.unsqueeze(0).unsqueeze(2).unsqueeze(3)
# #         avgout = torch.mean(x, dim=1, keepdim=True)
# #         maxout, _ = torch.max(x, dim=1, keepdim=True)
# #         out = torch.cat([avgout, maxout], dim=1)
# #         out = self.conv2d(out)
# #         weight=torch.mul(weight_bn, out)
# #         #return out
# #         return weight*x
# # class CoordAtt(nn.Module):
# #
# #     def __init__(self, in_channels, out_channels, reduction=32):
# #         super(CoordAtt, self).__init__()
# #         self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
# #         self.bn2 = nn.BatchNorm2d(in_channels)
# #         self.act1 = h_swish()
# #         self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
# #         self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
# #
# #     def forward(self, x):
# #         x=self.bn2(x)
# #         weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
# #         x = x.permute(0, 2, 3, 1).contiguous()
# #         x = torch.mul(weight_bn, x)
# #         x = x.permute(0, 3, 1, 2).contiguous()
# #         res=x
# #         x_h, x_w = self.pool_h(x), self.pool_w(x)
# #         out_h = self.conv2(x_h)
# #         out_w = self.conv3(x_w)
# #         w=torch.sigmoid_(out_h+out_w)
# #         return w*res
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
#         y = x.clone()
#         weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
#         x = x.permute(0, 2, 3, 1).contiguous()
#         x = torch.mul(weight_bn, x)
#         x = x.permute(0, 3, 1, 2).contiguous()
#         avgout = torch.mean(x, dim=1, keepdim=True)
#         maxout, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.conv2d(out)
#         weight = self.act(x*out)
#         # weight2=self.act(out)
#         return weight * res
#
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
#     def forward(self, x):
#         x=self.conv(x)
#         x=self.bn(x)
#         #NAM
#         weight_bn=self.bn.weight.data.abs()/torch.sum(self.bn.weight.data.abs())
#         y = x.permute(0, 2, 3, 1).contiguous()
#         y = torch.mul(weight_bn, y)
#         y = y.permute(0, 3, 1, 2).contiguous()
#
#         #CA
#         n, c, H, W = x.shape
#         feature=self.conv0(x)
#         x_h, x_w = self.pool_h(feature), self.pool_w(feature).permute(0, 1, 3, 2)
#         x_cat = torch.cat([x_h, x_w], dim=2)
#         out = self.act1(self.bn1(self.conv1(x_cat)))
#         x_h, x_w = torch.split(out, [H, W], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#         out_h = self.conv2(x_h).sigmoid()
#         out_w = self.conv3(x_w).sigmoid()
#
#         return x*out_w*out_h*torch.sigmoid(y)
#
# # class GSConv(nn.Module):
# #     # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
# #     def __init__(self, c1, c2, k=1, s=1,g=1, act=True):
# #         super().__init__()
# #         self.l_conv = ConvModule(
# #             c1,
# #             c2,
# #             1,
# #             inplace=False)
# #         self.conv1=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=1)
# #         self.conv2=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5,stride=1,padding=2)
# #
# #     #improve1
# #     def forward(self, x):
# #        x=self.l_conv(x)
# #        n,c,h,w=x.data.size()
# #        x1, x2 = torch.split(x, c // 2, dim=1)
# #
# #        x1_1,x1_2=torch.split(x1,c//4,dim=1)
# #
# #        # 沿着第二个维度求平均值，即在c维度上进行平均池化
# #        mean_x1 = torch.mean(x1_1, dim=1)
# #        # 将结果的维度由(n, w, h)变为(n, 1, w, h)
# #        mean_x1 = torch.unsqueeze(mean_x1, 1)
# #
# #        # 沿着第二个维度求平均值，即在c维度上进行平均池化
# #        mean_x2 = torch.mean(x1_2, dim=1)
# #        # 将结果的维度由(n, w, h)变为(n, 1, w, h)
# #        mean_x2 = torch.unsqueeze(mean_x2, 1)
# #
# #        weight1=self.conv1(mean_x1).sigmoid()
# #        weight2=self.conv2(mean_x2).sigmoid()
# #
# #        x1_1=x1_1*weight1
# #        x1_2=x1_2*weight2
# #        x1=torch.cat((x1_1,x1_2),1)
# #        #shuffle1
# #        b1,n1,h1,w1=x1.data.size()
# #        b1_n=b1*n1//2
# #        y=x1.reshape(b1_n,2,h*w)
# #        y=y.permute(1,0,2)
# #        y=y.reshape(2,-1,n1//2,h,w)
# #        x1=torch.cat((y[0],y[1]),1)
# #
# #        x=torch.cat((x1,x2),1)
# #        # shuffle2
# #        b2, n2, h2, w2 = x.data.size()
# #        b2_n = b2 * n2 // 2
# #        z = x.reshape(b2_n, 2, h2 * w2)
# #        z = z.permute(1, 0, 2)
# #        z = z.reshape(2, -1, n2 // 2, h2, w2)
# #        x = torch.cat((z[0],z[1]), 1)
# #        return x
#
# class GSConv(nn.Module):
#     # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=1, s=1,g=1, act=True):
#         super().__init__()
#         self.l_conv = ConvModule(
#             c1,
#             c2,
#             1,
#             inplace=False)
#         self.conv3 = Conv(c2 // 2, c2 // 2, 3, 1, None, c2 // 2, act)
#         self.conv5 = Conv(c2 // 2, c2 // 2, 5, 1, None, c2 // 2, act)
#
#     #improve1
#     def forward(self, x):
#        x=self.l_conv(x)
#        n,c,h,w=x.data.size()
#        x1, x2 = torch.split(x, c // 2, dim=1)
#
#        x1_gc=self.conv3(x1)
#        x2_gc=self.conv5(x2)
#
#        x1=torch.cat((x1,x1_gc),dim=1)
#
#        #shuffle1
#        b1,n1,h1,w1=x1.data.size()
#        b1_n=b1*n1//2
#        y=x1.reshape(b1_n,2,h*w)
#        y=y.permute(1,0,2)
#        y=y.reshape(2,-1,n1//2,h,w)
#        x1=torch.cat((y[0],y[1]),1)
#
#        x2=torch.cat((x2,x2_gc),dim=1)
#        # shuffle2
#        b2, n2, h2, w2 = x2.data.size()
#        b2_n = b2 * n2 // 2
#        z = x2.reshape(b2_n, 2, h2 * w2)
#        z = z.permute(1, 0, 2)
#        z = z.reshape(2, -1, n2 // 2, h2, w2)
#        x2 = torch.cat((z[0],z[1]), 1)
#
#        x=x1+x2
#        return x
# class CoordNAM(nn.Module):
#
#     def __init__(self, in_channels, out_channels):
#         super(CoordNAM, self).__init__()
#         self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.conv0 = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0)
#         self.conv1 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(1)
#         self.act1 = h_swish()
#         self.conv2 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
#         self.conv3 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
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
#         spa=out_h*out_w
#
#         return x *torch.sigmoid(spa+y)
#
# if __name__=='__main__':
#     x=torch.randn(50,512,7,7)
#     #S = CoordNAM(in_channels=512, out_channels=256)
#     S=GSConv(512,256)
#     t=S(x)
#     print(t.size())
#     # ins_feat = x  # 当前实例特征tensor
#     # # 生成从-1到1的线性值
#     # x_range = torch.linspace(-1, 1, ins_feat.shape[-1], device=ins_feat.device)
#     # y_range = torch.linspace(-1, 1, ins_feat.shape[-2], device=ins_feat.device)
#     # y, x = torch.meshgrid(y_range, x_range)  # 生成二维坐标网格
#     # y = y.expand([ins_feat.shape[0], 1, -1, -1])  # 扩充到和ins_feat相同维度
#     # x = x.expand([ins_feat.shape[0], 1, -1, -1])
#     # coord_feat = torch.cat([x, y], 1)  # 位置特征
#     # ins_feat = torch.cat([ins_feat, coord_feat], 1)  # concatnate一起作为下一个卷积的输入
#     # # #S=CoordAtt(in_channels=512,out_channels=512)
#     # # S=ChannelSpatial(512)
#     # # t=S(input)
#     # # print(t.size())
#     # from PIL import Image
#     #
#     # # opening a  image
#     # image = Image.open(r"/home/duomeitinrfx/data/tangka_magic_instrument/coco/test2017/11.jpg")
#     # img_scale = (1000, 1300)
#     # max_long_edge = max(img_scale)
#     # max_short_edge = min(img_scale)
#     # scale_factor = min(max_long_edge / max(image.height, image.width),
#     #                    max_short_edge / min(image.height, image.width))
#     # scale_w = int(image.width * float(scale_factor) + 0.5)
#     # scale_h = int(image.height * float(scale_factor) + 0.5)
#     # print(scale_w)


# import os
# import torch
#
# # 文件夹路径
# folder_path = '/home/duomeitinrfx/users/pengxl/multi label class/法器txt'
#
# # 获取目录下所有txt文件，并按文件名排序
# txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")], key=lambda x: int(''.join(filter(str.isdigit, x))))
#
# # 初始化一个空列表，用于存储每个txt文件中的数据
# data_list = []
#
# # 循环读取每个txt文件
# for filename in txt_files:
#     print(filename)
#     file_path = os.path.join(folder_path, filename)
#     with open(file_path, 'r') as file:
#         data = file.read().split()
#         data = [float(num) for num in data]  # 将字符串转换为浮点数
#         data_list.append(data)
#
# # 将数据列表转换为PyTorch张量
# tensor = torch.tensor(data_list)
#
# # 检查张量的形状
# print(tensor.shape)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 给定的数据
data = {
    'α': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'AR': [45.7, 46.4, 46.8, 46.8, 46.7, 46.5, 46.5, 46.4, 46.5, 46.2, 45.5],
    'AP': [36.2, 36.7, 37.3, 37.2, 37, 36.8, 36.8, 36.6, 36.7, 36.4, 36.3]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 设置图形大小
plt.figure(figsize=(7, 5))

# 创建第一个轴的实例
ax1 = plt.gca()
ax1.plot(df['α'], df['AP'], 'r-o')
ax1.set_xlabel('α')
ax1.set_ylabel('AP', color='r')
ax1.tick_params('y', colors='r')
ax1.set_ylim(36, 37.6)  # 设定 y 轴范围

# 创建第二个轴的实例
ax2 = ax1.twinx()
ax2.plot(df['α'], df['AR'], 'b-o')
ax2.set_ylabel('AR', color='b')
ax2.tick_params('y', colors='b')
ax2.set_ylim(45.4, 47)  # 设定 y 轴范围

# 设置标题

# 设置网格线仅在 x 轴方向
ax1.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
ax2.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)

# 调整刻度间隔
ax1.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax2.yaxis.set_major_locator(plt.MultipleLocator(0.2))

# 展示图形
plt.show()

