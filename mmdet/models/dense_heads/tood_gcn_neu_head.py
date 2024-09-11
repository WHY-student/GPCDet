# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init,kaiming_init
from mmcv.ops import deform_conv2d
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, distance2bbox,
                        images_to_levels, multi_apply, reduce_mean, unmap,build_sampler)
from mmdet.core.utils import filter_scores_and_topk
from mmdet.models.utils import sigmoid_geometric_mean
from ..builder import HEADS, build_loss
from .atss_neu_head  import ATSSNeuHead
import torch.nn.init as init
import os


class TaskDecomposition(nn.Module):
    """Task decomposition module in task-aligned predictor of TOOD.

    Args:
        feat_channels (int): Number of feature channels in TOOD head.
        stacked_convs (int): Number of conv layers in TOOD head.
        la_down_rate (int): Downsample rate of layer attention.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
    """

    def __init__(self,
                 feat_channels,
                 stacked_convs,
                 la_down_rate=8,
                 conv_cfg=None,
                 norm_cfg=None):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.norm_cfg = norm_cfg
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.in_channels // la_down_rate,
                self.stacked_convs,
                1,
                padding=0), nn.Sigmoid())

        self.reduction_conv = ConvModule(
            self.in_channels,
            self.feat_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=norm_cfg is None)

    def init_weights(self):
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.layer_attention(avg_feat)

        # here we first compute the product between layer attention weight and
        # conv weight, and then compute the convolution between new conv weight
        # and feature map, in order to save memory and FLOPs.
        conv_weight = weight.reshape(
            b, 1, self.stacked_convs,
            1) * self.reduction_conv.conv.weight.reshape(
                1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels,
                                          self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h,
                                                    w)
        if self.norm_cfg is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.activate(feat)

        return feat
class ClsTaskDecomposition(nn.Module):
    """Task decomposition module in task-aligned predictor of TOOD.

    Args:
        feat_channels (int): Number of feature channels in TOOD head.
        stacked_convs (int): Number of conv layers in TOOD head.
        la_down_rate (int): Downsample rate of layer attention.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
    """

    def __init__(self,
                 feat_channels,
                 stacked_convs,
                 la_down_rate=8,
                 conv_cfg=None,
                 norm_cfg=None):
        super(ClsTaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.norm_cfg = norm_cfg
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.in_channels // la_down_rate,
                self.stacked_convs,
                1,
                padding=0), nn.Sigmoid())

        self.reduction_conv = ConvModule(
            self.in_channels,
            self.feat_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=norm_cfg is None)

    def init_weights(self):
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.layer_attention(avg_feat)

        # here we first compute the product between layer attention weight and
        # conv weight, and then compute the convolution between new conv weight
        # and feature map, in order to save memory and FLOPs.
        conv_weight = weight.reshape(
            b, 1, self.stacked_convs,
            1) * self.reduction_conv.conv.weight.reshape(
                1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels,
                                          self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h,
                                                    w)
        if self.norm_cfg is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.activate(feat)

        return feat
class ResnetFeature(nn.Module):
    def __init__(self):
        super().__init__()
    def get_feature(self):
        # 文件夹路径
        folder_path = '/home/duomeitinrfx/users/pengxl/mmdetection/法器txt'

        # 获取目录下所有txt文件，并按文件名排序
        txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")],
                           key=lambda x: int(''.join(filter(str.isdigit, x))))

        # 初始化一个空列表，用于存储每个txt文件中的数据
        data_list = []

        # 循环读取每个txt文件
        for filename in txt_files:
            print(filename)
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = file.read().split()
                data = [float(num) for num in data]  # 将字符串转换为浮点数
                data_list.append(data)
        # 将数据列表转换为PyTorch张量
        feature = torch.tensor(data_list)
        # 步骤1: 找到最小值和最大值
        min_values = feature.min(dim=0).values
        max_values = feature.max(dim=0).values

        # 步骤2: 进行归一化
        feature = (feature - min_values) / (max_values - min_values)
        return feature


class GraphConvolution(nn.Module):


    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))  # W
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.weight.size(1))
    #     self.weight.data.uniform_(-stdv, stdv)
    #     if self.bias is not None:
    #         self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters(self):
        init.kaiming_normal_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input, adj,device):
        self.weight.to(device)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class ADJ(nn.Module):
    def __init__(self, num_classes, t,w=0.25):
        super(ADJ, self).__init__()
        self.num_classes = num_classes
        self.t = t
        self.w=w

    def gen_A(self):
        pass

    def gen_adj(self,device):  # 计算DAD
        _adj = np.array(
            [[1.0, 0.715, 0.21, 0.122, 0.448, 0.092, 0.609, 0.083, 0.264, 0.295, 0.144, 0.04, 0.013, 0.026, 0.178,
              0.16],
             [0.321, 1.0, 0.256, 0.116, 0.449, 0.096, 0.309, 0.091, 0.181, 0.177, 0.072, 0.037, 0.021, 0.047, 0.194,
              0.092],
             [0.291, 0.79, 1.0, 0.223, 0.473, 0.112, 0.314, 0.086, 0.168, 0.161, 0.059, 0.044, 0.024, 0.069, 0.25,
              0.086],
             [0.351, 0.748, 0.466, 1.0, 0.511, 0.183, 0.427, 0.187, 0.16, 0.153, 0.073, 0.034, 0.046, 0.118, 0.309,
              0.103],
             [0.341, 0.761, 0.26, 0.135, 1.0, 0.179, 0.401, 0.131, 0.173, 0.182, 0.074, 0.045, 0.037, 0.069, 0.228,
              0.096],
             [0.352, 0.814, 0.307, 0.241, 0.894, 1.0, 0.442, 0.151, 0.176, 0.141, 0.065, 0.045, 0.08, 0.06, 0.352,
              0.085],
             [0.669, 0.758, 0.25, 0.163, 0.579, 0.128, 1.0, 0.118, 0.242, 0.263, 0.122, 0.041, 0.029, 0.049, 0.228,
              0.123],
             [0.238, 0.581, 0.177, 0.185, 0.491, 0.113, 0.306, 1.0, 0.109, 0.249, 0.049, 0.026, 0.083, 0.132, 0.404,
              0.226],
             [0.533, 0.813, 0.245, 0.112, 0.459, 0.093, 0.445, 0.077, 1.0, 0.285, 0.12, 0.032, 0.021, 0.027, 0.157,
              0.165],
             [0.467, 0.623, 0.184, 0.084, 0.379, 0.059, 0.379, 0.138, 0.224, 1.0, 0.245, 0.056, 0.002, 0.017, 0.23,
              0.159],
             [0.574, 0.637, 0.168, 0.1, 0.389, 0.068, 0.442, 0.068, 0.237, 0.616, 1.0, 0.063, 0.0, 0.026, 0.095, 0.2],
             [0.385, 0.795, 0.308, 0.115, 0.577, 0.115, 0.359, 0.09, 0.154, 0.346, 0.154, 1.0, 0.013, 0.0, 0.244,
              0.077],
             [0.227, 0.795, 0.295, 0.273, 0.841, 0.364, 0.455, 0.5, 0.182, 0.023, 0.0, 0.023, 1.0, 0.25, 0.341, 0.045],
             [0.19, 0.752, 0.362, 0.295, 0.657, 0.114, 0.324, 0.333, 0.095, 0.076, 0.048, 0.0, 0.105, 1.0, 0.8, 0.267],
             [0.255, 0.62, 0.259, 0.153, 0.429, 0.132, 0.297, 0.202, 0.112, 0.208, 0.034, 0.036, 0.028, 0.159, 1.0,
              0.216],
             [0.44, 0.564, 0.171, 0.098, 0.349, 0.062, 0.309, 0.218, 0.225, 0.276, 0.138, 0.022, 0.007, 0.102, 0.415,
              1.0]])
        # ##有些共现，有噪声等情况，采用二值的方式进行解决
        _adj[_adj < self.t] = 0
        _adj[_adj >= self.t] = 1

        # 然后平滑化
        _adj = _adj * self.w / (_adj.sum(0, keepdims=True) + 1e-6)  # sum(0,keepdims=True)求数组每行的和
        _adj = _adj + np.identity(self.num_classes, np.int)  # np.id是一个单位矩阵  相当于得到A拔
        A=Parameter(torch.from_numpy(_adj).float())
        A=A.to(device)
        D = torch.pow(A.sum(1).float(), -0.5)    # A.sum(1)求度，然后D的-0.5次方
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj


@HEADS.register_module()
class TOODGCNNEUHead(ATSSNeuHead):
    """TOODHead used in `TOOD: Task-aligned One-stage Object Detection.

    <https://arxiv.org/abs/2108.07755>`_.

    TOOD uses Task-aligned head (T-head) and is optimized by Task Alignment
    Learning (TAL).

    Args:
        num_dcn (int): Number of deformable convolution in the head.
            Default: 0.
        anchor_type (str): If set to `anchor_free`, the head will use centers
            to regress bboxes. If set to `anchor_based`, the head will
            regress bboxes based on anchors. Default: `anchor_free`.
        initial_loss_cls (dict): Config of initial loss.

    Example:
        >>> self = TOODHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_dcn=0,
                 anchor_type='anchor_free',
                 t=0.7,# gcn中二值化的阈值
                 w=0.25,
                 reluw=0.2,
                 lossNeuWeight_fra=0.25,
                 initial_loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     activated=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_neu=None,
                 **kwargs):
        assert anchor_type in ['anchor_free', 'anchor_based']
        self.num_dcn = num_dcn
        self.anchor_type = anchor_type
        self.t=t
        self.w=w
        self.reluw=reluw
        self.lossNeuWeight_fra = lossNeuWeight_fra
        self.epoch = 0  # which would be update in SetEpochInfoHook!
        super(TOODGCNNEUHead, self).__init__(num_classes, in_channels, **kwargs)
        self.loss_bbox_neu = build_loss(loss_bbox_neu)
        if self.train_cfg:
            self.initial_epoch = self.train_cfg.initial_epoch
            self.initial_assigner = build_assigner(
                self.train_cfg.initial_assigner)
            self.initial_loss_cls = build_loss(initial_loss_cls)
            self.assigner = self.initial_assigner
            self.alignment_assigner = build_assigner(self.train_cfg.assigner)
            self.alpha = self.train_cfg.alpha
            self.beta = self.train_cfg.beta
            sampler_cfg = dict(type='PseudoSamplerNeu')
            self.sampler = build_sampler(sampler_cfg, context=self)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.inter_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            if i < self.num_dcn:
                conv_cfg = dict(type='DCNv2', deform_groups=4)
            else:
                conv_cfg = self.conv_cfg
            chn = self.in_channels if i == 0 else self.feat_channels
            self.inter_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.cls_decomp = ClsTaskDecomposition(self.feat_channels,
                                            self.stacked_convs,
                                            self.stacked_convs * 8,
                                            self.conv_cfg, self.norm_cfg)
        self.reg_decomp = TaskDecomposition(self.feat_channels,
                                            self.stacked_convs,
                                            self.stacked_convs * 8,
                                            self.conv_cfg, self.norm_cfg)

        self.tood_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,# 17(包括一个背景类)
            3,
            padding=1)
        self.tood_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)

        self.cls_prob_module = nn.Sequential(
            nn.Conv2d(self.feat_channels * self.stacked_convs,
                      self.feat_channels // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, 1, 3, padding=1))
        self.reg_offset_module = nn.Sequential(
            nn.Conv2d(self.feat_channels * self.stacked_convs,
                      self.feat_channels // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, 4 * 2, 3, padding=1))

        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])
        self.gcn1 = GraphConvolution(in_features=2048,out_features=1024,bias=False)
        self.gcn2 = GraphConvolution(in_features=1024,out_features=512,bias=False)
        self.gcn3 = GraphConvolution(in_features=512, out_features=256,bias=False)

        self.conv_cat =nn.Sequential(
            nn.Conv2d(
            288,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1))
        self.conv_w=nn.Conv2d(self.in_channels,1,1)
        self.gc_fc = nn.Linear(self.num_classes *256 , 256)
        self.weight_fc = nn.Linear(256, 256)
        #self.gcn4 = GraphConvolution(in_features=512, out_features=512)
        #self.gcn4 = GraphConvolution(in_features=256, out_features=256)
        # self.gcn3 = GraphConvolution(in_features=512, out_features=512)
        # self.gcn4=GraphConvolution(in_features=512,out_features=256)
        self.leakyrelu=nn.LeakyReLU(self.reluw)
        self.adj=ADJ(num_classes=self.num_classes,t=self.t,w=self.w)

        # resnet50获取特征
        fea=ResnetFeature()
        self.in_feat=fea.get_feature()
    def init_weights(self):
        """Initialize weights of the head."""
        bias_cls = bias_init_with_prob(0.01)
        for m in self.inter_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cls_prob_module:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_offset_module:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        for m in self.conv_cat:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.cls_prob_module[-1], std=0.01, bias=bias_cls)

        self.cls_decomp.init_weights()
        self.reg_decomp.init_weights()

        normal_init(self.tood_cls, std=0.01, bias=bias_cls)
        normal_init(self.tood_reg, std=0.01)

        init.xavier_uniform_(self.gc_fc.weight)
        init.constant_(self.gc_fc.bias, 0)
        init.xavier_uniform_(self.weight_fc.weight)
        init.constant_(self.weight_fc.bias, 0)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Decoded box for all scale levels,
                    each is a 4D-tensor, the channels number is
                    num_anchors * 4. In [tl_x, tl_y, br_x, br_y] format.
        """
        cls_scores = []
        bbox_preds = []
        for idx, (x, scale, stride) in enumerate(
                zip(feats, self.scales, self.prior_generator.strides)):
            b, c, h, w = x.shape
            # adj = self.adj.gen_adj(device=x.device).detach()  # 这里提前将DAD计算好
            # g_feat = self.gcn1(self.in_feat.to(x.device), adj, x.device)  #
            # g_feat = self.leakyrelu(g_feat)
            # g_feat = self.gcn2(g_feat, adj, x.device)
            # g_feat = self.leakyrelu(g_feat)
            # g_feat = self.gcn3(g_feat, adj, x.device)
            #
            # g_feat_flattened = g_feat.view(-1)
            # g_feat_flattened = self.gc_fc(g_feat_flattened)
            # g_feat_flattened = g_feat_flattened.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            # g_feat_flattened = g_feat_flattened.repeat(b, 1, h, w)
            # x = torch.cat([x, g_feat_flattened], dim=1)
            # x=self.conv_cat(x)


            anchor = self.prior_generator.single_level_grid_priors(
                (h, w), idx, device=x.device)
            anchor = torch.cat([anchor for _ in range(b)])
            # extract task interactive features
            inter_feats = []
            for inter_conv in self.inter_convs:
                x = inter_conv(x)
                inter_feats.append(x)
            feat = torch.cat(inter_feats, 1)

            # task decomposition
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat = self.cls_decomp(feat, avg_feat)  # n*256*h*w
            reg_feat = self.reg_decomp(feat, avg_feat)  # n*256*h*w

            #gcn
            adj = self.adj.gen_adj(device=cls_feat.device).detach()  # 这里提前将DAD计算好
            g_feat = self.gcn1(self.in_feat.to(cls_feat.device), adj, cls_feat.device)  #
            g_feat = self.leakyrelu(g_feat)
            g_feat = self.gcn2(g_feat, adj, cls_feat.device)
            g_feat = self.leakyrelu(g_feat)
            g_feat = self.gcn3(g_feat, adj, cls_feat.device)

            g_feat_flattened = g_feat.view(-1)
            g_feat_flattened=self.gc_fc(g_feat_flattened)
            g_feat_flattened = g_feat_flattened.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            g_feat_flattened = g_feat_flattened.repeat(b, 1, h, w)
            #cls_feat = torch.cat([cls_feat, g_feat_flattened], dim=1)
            # cls_logits = self.conv_cat(cls_feat)

            # g_feat=g_feat.transpose(0,1)
            # cls_logits=torch.matmul(cls_feat.view(b,c,-1).permute(0,2,1),g_feat)#用这个来代替分类
            # cls_logits=cls_logits.permute(0,2,1)
            # cls_logits=cls_logits.view(b,self.num_classes,h,w)

            # 特征加权方式gcn_fea*w+cnn_fea#
            #通过CNN特征获取权重
            pool_feature=F.adaptive_avg_pool2d(cls_feat,(1,1))
            pool_feature=pool_feature.view(pool_feature.size(0),-1)
            weights=torch.sigmoid(self.weight_fc(pool_feature))
            weights=weights.view(weights.size(0),weights.size(1),1,1)
            cls_feat=weights*g_feat_flattened+cls_feat
            #

            # #空间特征权重
            # spa_weight=torch.sigmoid(self.conv_w(cls_feat))
            # cls_feat=cls_feat+spa_weight*g_feat_flattened

            # # cls prediction and alignment
            cls_logits = self.tood_cls(cls_feat)  # n*16*h*w
            cls_prob = self.cls_prob_module(feat)  # n*1*h*w 用于偏移类
            cls_score = sigmoid_geometric_mean(cls_logits, cls_prob)

            # reg prediction and alignment
            if self.anchor_type == 'anchor_free':
                reg_dist = scale(self.tood_reg(reg_feat).exp()).float()
                reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, 4)
                reg_bbox = distance2bbox(
                    self.anchor_center(anchor) / stride[0],
                    reg_dist).reshape(b, h, w, 4).permute(0, 3, 1,
                                                          2)  # (b, c, h, w)
            elif self.anchor_type == 'anchor_based':
                reg_dist = scale(self.tood_reg(reg_feat)).float()
                reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, 4)
                reg_bbox = self.bbox_coder.decode(anchor, reg_dist).reshape(
                    b, h, w, 4).permute(0, 3, 1, 2) / stride[0]
            else:
                raise NotImplementedError(
                    f'Unknown anchor type: {self.anchor_type}.'
                    f'Please use `anchor_free` or `anchor_based`.')
            reg_offset = self.reg_offset_module(feat)
            bbox_pred = self.deform_sampling(reg_bbox.contiguous(),
                                             reg_offset.contiguous())

            # After deform_sampling, some boxes will become invalid (The
            # left-top point is at the right or bottom of the right-bottom
            # point), which will make the GIoULoss negative.
            invalid_bbox_idx = (bbox_pred[:, [0]] > bbox_pred[:, [2]]) | \
                               (bbox_pred[:, [1]] > bbox_pred[:, [3]])
            invalid_bbox_idx = invalid_bbox_idx.expand_as(bbox_pred)
            bbox_pred = torch.where(invalid_bbox_idx, reg_bbox, bbox_pred)

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return tuple(cls_scores), tuple(bbox_preds)

    def deform_sampling(self, feat, offset):
        """Sampling the feature x according to offset.

        Args:
            feat (Tensor): Feature
            offset (Tensor): Spatial offset for feature sampling
        """
        # it is an equivalent implementation of bilinear interpolation
        b, c, h, w = feat.shape
        weight = feat.new_ones(c, 1, 1, 1)
        y = deform_conv2d(feat, offset, weight, 1, 0, 1, c, c)
        return y

    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def loss_single(self, anchors, cls_score, bbox_pred, labels, label_weights,
                    labels_pos, labels_neu, bbox_targets, alignment_metrics, stride):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            alignment_metrics (Tensor): Alignment metrics with shape
                (N, num_total_anchors).
            stride (tuple[int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        labels_pos = labels_pos.reshape(-1)
        labels_neu = labels_neu.reshape(-1)
        alignment_metrics = alignment_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = labels if self.epoch < self.initial_epoch else (
            labels, alignment_metrics)
        cls_loss_func = self.initial_loss_cls \
            if self.epoch < self.initial_epoch else self.loss_cls

        loss_cls = cls_loss_func(
            cls_score, targets, label_weights, avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels_pos >= 0)
                    & (labels_pos < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]

            # regression loss
            pos_bbox_weight = self.centerness_target(
                pos_anchors, pos_bbox_targets
            ) if self.epoch < self.initial_epoch else alignment_metrics[
                pos_inds]

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)

        neu_inds = ((labels_neu >= 0)
                    & (labels_neu < bg_class_ind)).nonzero().squeeze(1)

        if len(neu_inds) > 0:
            neu_bbox_targets = bbox_targets[neu_inds]
            neu_bbox_pred = bbox_pred[neu_inds]
            neu_anchors = anchors[neu_inds]

            neu_decode_bbox_pred = neu_bbox_pred
            neu_decode_bbox_targets = neu_bbox_targets / stride[0]

            # regression loss
            neu_bbox_weight = self.centerness_target(
                neu_anchors, neu_bbox_targets
            ) if self.epoch < self.initial_epoch else alignment_metrics[
                neu_inds]

            loss_bbox_neu = self.loss_bbox_neu(
                neu_decode_bbox_pred,
                neu_decode_bbox_targets,
                weight=neu_bbox_weight,
                avg_factor=1.0)
        else:
            loss_bbox_neu = bbox_pred.sum() * 0
            neu_bbox_weight = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_bbox_neu, alignment_metrics.sum(
        ), pos_bbox_weight.sum(), neu_bbox_weight.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Decoded box for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1)
        flatten_bbox_preds = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) * stride[0]
            for bbox_pred, stride in zip(bbox_preds,
                                         self.prior_generator.strides)
        ], 1)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bbox_preds,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        (anchor_list, labels_list, label_weights_list,
         labels_list_pos, labels_list_neu,
         bbox_targets_list,
         alignment_metrics_list) = cls_reg_targets

        losses_cls, losses_bbox, losses_bbox_neu, \
            cls_avg_factors, bbox_avg_factors, bbox_avg_factors_neu = multi_apply(
            self.loss_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            labels_list_pos,
            labels_list_neu,
            bbox_targets_list,
            alignment_metrics_list,
            self.prior_generator.strides)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        bbox_avg_factors_neu = reduce_mean(
            sum(bbox_avg_factors_neu)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        losses_bbox2 = list(map(lambda x: x / bbox_avg_factors_neu, losses_bbox_neu))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, losses_bbox2=losses_bbox2)

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """

        cfg = self.test_cfg if cfg is None else cfg
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for cls_score, bbox_pred, priors, stride in zip(
                cls_score_list, bbox_pred_list, mlvl_priors,
                self.prior_generator.strides):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4) * stride[0]
            scores = cls_score.permute(1, 2,
                                       0).reshape(-1, self.cls_out_channels)

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bboxes = filtered_results['bbox_pred']

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, None, **kwargs)

    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            cls_scores (Tensor): Classification predictions of images,
                a 3D-Tensor with shape [num_imgs, num_priors, num_classes].
            bbox_preds (Tensor): Decoded bboxes predictions of one image,
                a 3D-Tensor with shape [num_imgs, num_priors, 4] in [tl_x,
                tl_y, br_x, br_y] format.
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: a tuple containing learning targets.

                - anchors_list (list[list[Tensor]]): Anchors of each level.
                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - norm_alignment_metrics_list (list[Tensor]): Normalized
                  alignment metrics of each level.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]  # 每层的anchor数量 一个list
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):  # 将一个batchsize的anchorconcat成一个tensor
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        # anchor_list: list(b * [-1, 4])

        if self.epoch < self.initial_epoch:
            (all_anchors, all_labels, all_label_weights, all_labels_neu, all_bbox_targets,
             all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
                super()._get_target_single,
                anchor_list,
                valid_flag_list,
                num_level_anchors_list,
                gt_bboxes_list,
                gt_bboxes_ignore_list,
                gt_labels_list,
                img_metas,
                label_channels=label_channels,
                unmap_outputs=unmap_outputs)
            all_assign_metrics = [
                weight[..., 0] for weight in all_bbox_weights
            ]
            all_labels_pos = all_labels
        else:
            (all_anchors, all_labels, all_label_weights, all_labels_pos, all_labels_neu, all_bbox_targets,
             all_assign_metrics) = multi_apply(
                self._get_target_single,
                cls_scores,
                bbox_preds,
                anchor_list,
                valid_flag_list,
                gt_bboxes_list,
                gt_bboxes_ignore_list,
                gt_labels_list,
                img_metas,
                label_channels=label_channels,
                unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        if any([labels is None for labels in all_labels_neu]):
            return None
        if any([labels is None for labels in all_labels_pos]):
            return None
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        labels_list_pos = images_to_levels(all_labels_pos, num_level_anchors)
        labels_list_neu = images_to_levels(all_labels_neu, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        norm_alignment_metrics_list = images_to_levels(all_assign_metrics,
                                                       num_level_anchors)

        return (anchors_list, labels_list, label_weights_list, labels_list_pos,
                labels_list_neu, bbox_targets_list, norm_alignment_metrics_list)

    def _get_target_single(self,
                           cls_scores,  # 预测的分数 （n,16） n=所有层预测的
                           bbox_preds,  # 预测的边界框(n,4)
                           flat_anchors,  # anchor 这里的anchor是将所有层的拼接到了一个维度(n,4)
                           valid_flags,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            cls_scores (list(Tensor)): Box scores for each image.
            bbox_preds (list(Tensor)): Box energies / deltas for each image.
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                norm_alignment_metrics (Tensor): Normalized alignment metrics
                    of all priors in the image with shape (N,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,  # tensor(n)
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        assign_result = self.alignment_assigner.assign(
            cls_scores[inside_flags, :], bbox_preds[inside_flags, :], anchors,
            gt_bboxes, gt_bboxes_ignore, gt_labels, self.alpha, self.beta)
        assign_ious = assign_result.max_overlaps
        assign_ious2 = assign_result.max_overlaps2
        assign_metrics = assign_result.assign_metrics
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,),
                                  self.num_classes,
                                  dtype=torch.long)
        labels_pos = anchors.new_full((num_valid_anchors,),
                                      self.num_classes,
                                      dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        norm_alignment_metrics = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        neu_inds = sampling_result.neu_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
                labels_pos[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
                labels_pos[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        labelsNeu = anchors.new_full((num_valid_anchors,),
                                     self.num_classes,
                                     dtype=torch.long)
        if len(neu_inds) > 0:
            neu_bbox_targets = sampling_result.neu_gt_bboxes
            bbox_targets[neu_inds, :] = neu_bbox_targets

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[neu_inds] = 0
                labelsNeu[neu_inds] = 0
            else:
                labels[neu_inds] = gt_labels[
                    sampling_result.neu_assigned_gt_inds]
                labelsNeu[neu_inds] = gt_labels[
                    sampling_result.neu_assigned_gt_inds]
            label_weights[neu_inds] = self.lossNeuWeight_fra * 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds ==
                                     gt_inds]
            pos_alignment_metrics = assign_metrics[gt_class_inds]
            pos_ious = assign_ious[gt_class_inds]
            pos_norm_alignment_metrics = pos_alignment_metrics / (
                    pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
            norm_alignment_metrics[gt_class_inds] = pos_norm_alignment_metrics

        # neu
        class_assigned_gt_inds_neu = torch.unique(
            sampling_result.neu_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds_neu:
            gt_class_inds_neu = neu_inds[sampling_result.neu_assigned_gt_inds ==
                                         gt_inds]
            neu_alignment_metrics = assign_metrics[gt_class_inds_neu]
            neu_ious = assign_ious2[gt_class_inds_neu]
            neu_norm_alignment_metrics = neu_alignment_metrics / (
                    neu_alignment_metrics.max() + 10e-8) * neu_ious.max()
            norm_alignment_metrics[gt_class_inds_neu] = neu_norm_alignment_metrics

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            norm_alignment_metrics = unmap(norm_alignment_metrics,
                                           num_total_anchors, inside_flags)
            labels_pos = unmap(
                labels_pos, num_total_anchors, inside_flags, fill=self.num_classes)
            labelsNeu = unmap(
                labelsNeu, num_total_anchors, inside_flags, fill=self.num_classes)

        return (anchors, labels, label_weights, labels_pos, labelsNeu, bbox_targets,
                norm_alignment_metrics)

