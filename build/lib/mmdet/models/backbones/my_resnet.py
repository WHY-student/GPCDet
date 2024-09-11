# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner.base_module import ModuleList, Sequential
from ..builder import BACKBONES
from ..utils import ResLayer


# inplane是输入的通道数,plane是输出的通道数,expansion是对输出通道数的倍乘.
class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 with_cp=False, # 是否对残差块进行0初始化
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):

        super(Bottleneck, self).__init__(init_cfg)
        self.inplanes = inplanes
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        # block中第一个卷积不做改变
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=1,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False

        # block中的第二个卷积将卷积核设置为由 3-> 7, stride也改成1，单独坐下采样，同时padding也需要进行更改
        self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=7,
                stride=1,
                padding=3,
                dilation=dilation,
                bias=False)
        self.add_module(self.norm2_name, norm2)

        # block中的第三个卷积不做更改
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)
        self.con_res=build_conv_layer(
            conv_cfg,
            inplanes,
            planes*self.expansion,
            kernel_size=1,
            bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            # 减少激活函数的使用
            #out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)
            identity=self.con_res(identity)
            identity=self.norm3(identity)
            out += identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        # 减少激活函数的使用
        # out = self.relu(out)
        return out


@BACKBONES.register_module()
class MyResNet(BaseModule):
    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth, # 网络深度
                 in_channels=3, # 输入图像的通道
                 stem_channels=None, # 主干卷积层的channel数，默认等于base_channel
                 base_channels=96, # base_channels进行更改由 64->96
                 num_stages=4, # stage数量
                 strides=(1, 1, 1, 1),  # 每个残差块的stride都变成了1

                 out_indices=(0, 1, 2, 3), # 输出特征图的索引，每个stage对应一个
                 avg_down=False,   # 是否使用平均池化代替stride为2的卷积操作进行下采样
                 frozen_stages=-1,# 冻结层数，-1表示不冻结
                 conv_cfg=None,  # 构建卷积层的配置
                 norm_cfg=dict(type='BN', requires_grad=True), # 构建归一化层的配置
                 norm_eval=True,
                 with_cp=False,
                 zero_init_residual=True,  # 是否对残差块进行0初始化
                 pretrained=None, # 预训练模型（若指定会自动调用init_cfg）
                 init_cfg=None):  # 指定预训练模型
        super(MyResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides

        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels
        # 构建stem部分
        self._make_stem_layer(in_channels, stem_channels)
        # 构建残差部分
        self.res_layers = []
        self.plane_list=[96,192,384,768]

        self.downsample_layers = ModuleList()

        self.downsample_layers.append(self.stem)
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = 1
            if i >= 1:
                downsample_layer=nn.Sequential(
                    nn.BatchNorm2d(self.plane_list[i]*2),
                    nn.Conv2d(
                        self.plane_list[i]*2,
                        self.plane_list[i],
                        kernel_size=2,
                        stride=2))
                self.downsample_layers.append(downsample_layer)
            planes = self.plane_list[i]
            self.inplanes=self.plane_list[i]
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        #
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        self.stem=nn.Sequential(
            nn.Conv2d(
                in_channels,
                stem_channels,
                kernel_size=4,
                stride=4),
            build_norm_layer(self.norm_cfg,stem_channels)[1],
        )

    # 固定权重，需要两个步骤：1. 设置 eval 模式；2. requires_grad=False
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            # 固定 stem 权重
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False
        # 固定 stage 权重
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            x=self.downsample_layers[i](x)
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

# 上述过程只是完成了resnet50的初始化。在评估模型时，可以直接使用冻结好的resnet50；
# 但是在train模式下，resnet50的所有层进入训练模式，白固定了，因此需要重新写train方法。
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(MyResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

