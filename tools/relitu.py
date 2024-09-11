#coding: utf-8
import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import os
import torch
import torch.nn as nn
import warnings

from mmcv.ops import RoIAlign, RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, init_detector
from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmdet.models.dense_heads import *

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def draw_feature_map(model, img_path, save_dir):
    '''
    :param model: 加载了参数的模型
    :param img_path: 测试图像的文件路径
    :param save_dir: 保存生成图像的文件夹
    :return:
    '''
    img = mmcv.imread(img_path)
    modeltype = str(type(model)).split('.')[-1].split('\'')[0]
    model.eval()
    model.draw_heatmap = True
    featuremaps = inference_detector(model, img)
    i=0
    for featuremap in featuremaps:
        heatmap = featuremap_2_heatmap(featuremap)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
        cv2.imwrite(os.path.join(save_dir,'featuremap_'+str(i)+'.png'), superimposed_img)  # 将图像保存到硬盘
        i=i+1


from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--img',default="/home/duomeitinrfx/data/tangka_magic_instrument/VOCdevkit/VOC2007/JPEGImages/995.jpg", help='Image file')
    parser.add_argument('--save_dir',default="/home/duomeitinrfx/users/pengxl/mmdetection/tools/work_dirs_0.8_0.2/1000_1300_dcn/tood_fusion/static/0.383_0.413_lr0.5/", help='Dir to save heatmap')
    parser.add_argument('--config',default="/home/duomeitinrfx/users/pengxl/mmdetection/tools/work_dirs_0.8_0.2/1000_1300_dcn/tood_fusion/static/0.383_0.413_lr0.5/tood_fpn.py", help='Config file')
    parser.add_argument('--checkpoint',default="/home/duomeitinrfx/users/pengxl/mmdetection/tools/work_dirs_0.8_0.2/1000_1300_dcn/tood_fusion/static/0.383_0.413_lr0.5/best_bbox_mAP_epoch_18.pth", help='Checkpoint file')
    parser.add_argument('--device', default='cuda:3', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    draw_feature_map(model,args.img,args.save_dir)

if __name__ == '__main__':
    main()