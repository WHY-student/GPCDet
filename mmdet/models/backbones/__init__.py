# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .swin import SwinTransformer
from .trident_resnet import TridentResNet
from  .my_resnet import MyResNet
from .resnet_attention import ResNet_ATTENTION
from .resnet_sac_attention import ResNet_SAC_ATTENTION
from .resnext_attention import ResNeXt_Attention
from .resnet_rfp import ResNet_RFP
__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'EfficientNet','MyResNet','ResNet_ATTENTION','ResNet_SAC_ATTENTION',
    'ResNeXt_Attention','ResNet_RFP'
]
