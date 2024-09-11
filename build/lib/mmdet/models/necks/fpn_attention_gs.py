# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import  build_norm_layer
from ..builder import NECKS
from ..utils import CALayer,MS_CAM,CBAMLayer
from ..utils import CA,ShuffleAttention,CrissCrossAttention,NAMAttention,SKAttention,SimAM,S2Attention,GAMAttention,BAMBlock,GlobalContextBlock
from ..utils.asff import ASFF
from ..utils.gsconv import GSConv

@NECKS.register_module()
class FPN_ATTENTION_GS(BaseModule):
    """
# 这是一个例子
    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels, # 每个尺度的输入通道数 list格式
                 out_channels, #输出通道
                 num_outs, #输出的尺度数,这里如果的输出数与in_channels的长度不一致，fpn会进行下采样，这个可以用来当做双阶段第一阶段的RPN阶段
                 start_level=0, #  用于构建FPN的起始输入主干水平的索引。默认值：0
                 end_level=-1, # 用于构建FPN的末端输入主干水平（独占）的索引。默认值：-1，表示最后一个级别。
                 attention=None,# attetion的模块，默认不使用有"CA","MS_CAM","CBAM"
                 # 值可为(bool | str)
                 # 如果为 bool，则决定是否在原始特征图的顶部添加conv layers。默认为“False”。
                 # 如果为 True，则等效于add_extra_convs=“on_input”。
                 # 如果是 str，则指定额外 conv 的源特征映射。仅允许以下选项
                 # ’on_input’: Last feat map of neck inputs (i.e. backbone feature).
                 # ’on_lateral’: Last feature map after lateral convs.
                 # ’on_output’: The last output feature map after fpn convs.
                 add_extra_convs=False,
                 asff=False,
                 res=False,
                 relu_before_extra_convs=False,  # 在extra convs之前是否使用relu
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None, # Config dict for activation layer in ConvModule
                 upsample_cfg=dict(mode='nearest'),# 上采样的方式
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FPN_ATTENTION_GS, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.attention_name=attention
        self.res=res
        self.asff=asff
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList() #  C4/5/6-C4/5/6 的1*1卷积
        self.fpn_convs = nn.ModuleList()    # C4/5/6-P4/5/6 的3*3卷积
        self.res_convs=nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level): # start level从1开始到3
            if i==self.start_level:
                l_conv=GSConv(c1=in_channels[i],c2=out_channels)
            else:
                l_conv = ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            res_conv=ConvModule(out_channels, out_channels, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
                       inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.res_convs.append(res_conv)

        self.attentions=nn.ModuleList()
        self.norm1_name, norm1 = build_norm_layer(dict(type='GN', num_groups=32, requires_grad=True),out_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(dict(type='GN', num_groups=32, requires_grad=True), out_channels, postfix=1)
        self.norm3_name, norm3 = build_norm_layer(dict(type='GN', num_groups=32, requires_grad=True), out_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.add_module(self.norm2_name, norm2)
        self.add_module(self.norm3_name, norm3)

        self.asff_toplevel = None
        self.asff_midlevel = None
        self.asff_bottomlevel = None
        if self.asff:
            self.asff_toplevel = ASFF(level=0)
            self.asff_midlevel = ASFF(level=1)
            self.asff_bottomlevel = ASFF(level=2)

        if self.attention_name=='CALayer':
            for i in range(self.start_level, self.backbone_end_level):
                attention_=CALayer(out_channels)
                self.attentions.append(attention_)
        elif self.attention_name=='MS_CAM':
            for i in range(self.start_level, self.backbone_end_level):
                attention_=MS_CAM(out_channels)
                self.attentions.append(attention_)
        elif self.attention_name=='CBAM':
            for i in range(self.start_level, self.backbone_end_level):
                attention_=CBAMLayer(out_channels)
                self.attentions.append(attention_)
        elif self.attention_name=='CA': # Coordinate Attention
            for i in range(self.start_level, self.backbone_end_level):
                attention_=CA(inp=out_channels,oup=out_channels)
                self.attentions.append(attention_)
        elif self.attention_name=='ShuffleAttention':
            for i in range(self.start_level, self.backbone_end_level):
                attention_=ShuffleAttention(out_channels)
                self.attentions.append(attention_)
        elif self.attention_name=='CrissCrossAttention':
            for i in range(self.start_level, self.backbone_end_level):
                attention_=CrissCrossAttention(out_channels)
                self.attentions.append(attention_)
        elif self.attention_name=='NAMAttention':
            for i in range(self.start_level, self.backbone_end_level):
                attention_ = NAMAttention(out_channels)
                self.attentions.append(attention_)
        elif self.attention_name=='SKAttention':
            for i in range(self.start_level, self.backbone_end_level):
                attention_ = SKAttention(out_channels)
                self.attentions.append(attention_)
        elif self.attention_name=='SimAM':
            for i in range(self.start_level, self.backbone_end_level):
                attention_ = SimAM()
                self.attentions.append(attention_)
        elif self.attention_name=='S2Attention':
            for i in range(self.start_level, self.backbone_end_level):
                attention_ = S2Attention(out_channels)
                self.attentions.append(attention_)
        elif self.attention_name=='GAMAttention':
            for i in range(self.start_level, self.backbone_end_level):
                attention_ = GAMAttention(c1=out_channels,c2=out_channels)
                self.attentions.append(attention_)
        elif self.attention_name=='BAM':
            for i in range(self.start_level, self.backbone_end_level):
                attention_ = BAMBlock(channel=out_channels)
                self.attentions.append(attention_)
        elif self.attention_name=='GlobalContextBlock':
            for i in range(self.start_level, self.backbone_end_level):
                attention_ = GlobalContextBlock(out_channels)
                self.attentions.append(attention_)
        # add extra conv layers (e.g., RetinaNet)
        # 计算需要几个额外的特征图
        extra_levels = num_outs - self.backbone_end_level + self.start_level # extra_levels=2
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1] # n_channels=inchannel[3]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                res_conv = ConvModule(out_channels, out_channels, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,act_cfg=act_cfg,inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
                self.res_convs.append(res_conv)

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm3_name)

    @auto_fp16()
    def forward(self, inputs): # 这里输入是backbone传来的4层的张量，只用到后三层
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # 首先经过1*1卷积得到将C3-C5特征图变成相同的通道数
        laterals = [
            lateral_conv(inputs[i + self.start_level])for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        if self.attention_name!=None:
            if self.res==False:
                for i in range(len(laterals)):
                    laterals[i]=self.attentions[i](laterals[i])
            else:
                lat_res=[
                    res_conv(laterals[i])
                    for i,res_conv in enumerate(self.res_convs)
                ]
                for i in range(len(lat_res)):
                    laterals[i]=self.attentions[i](lat_res[i])+laterals[i]
                laterals[0]=self.norm1(laterals[0])
                laterals[1]=self.norm2(laterals[1])
                laterals[2]=self.norm3(laterals[2])
        #  向上插值在相加
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
        if self.asff:
            laterals[0] = self.asff_bottomlevel(laterals[2], laterals[1], laterals[0])
            laterals[1]=self.asff_midlevel(laterals[2], laterals[1], laterals[0])
            laterals[2]=self.asff_toplevel(laterals[2], laterals[1], laterals[0])
        # 经过3*3卷积得到P3-P5
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))


            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1] # 直接从backbone最后输出当做输入
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
