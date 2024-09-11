# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    # parser.add_argument('--img',default="/home/duomeitinrfx/data/tangka_magic_instrument/update/VOCdevkit/VOC2007/JPEGImages/160.jpg" ,help='Image file')
    # parser.add_argument('--config',default='/home/duomeitinrfx/users/pengxl/mmdetection/tools/work_dirs_0.8_0.2/1000_1300/tood/tood/tood_fpn.py', help='Config file')
    # parser.add_argument('--checkpoint', default='/home/duomeitinrfx/users/pengxl/mmdetection/tools/work_dirs_0.8_0.2/1000_1300/tood/tood/best_bbox_mAP_epoch_19.pth',help='Checkpoint file')
    # parser.add_argument('--out-file', default='/home/duomeitinrfx/users/pengxl/mmdetection/tools/work_dirs_0.8_0.2/1000_1300/tood/tood/160.jpg', help='Path to output file')
    parser.add_argument('--img',
                        default="/home/duomeitinrfx/data/tangka_magic_instrument/update/coco_0.8_0.2/val/2756.jpg",
                        help='Image file')
    parser.add_argument('--config',
                        default='/home/duomeitinrfx/users/pengxl/mmdetection/tools/work_dirs_0.8_0.2/1000_1300_dcn/prio_gcn_ca/tood/tood_lr0.6_5/tood_fpn.py',
                        help='Config file')
    parser.add_argument('--checkpoint',
                        default='/home/duomeitinrfx/users/pengxl/mmdetection/tools/work_dirs_0.8_0.2/1000_1300_dcn/prio_gcn_ca/tood/tood_lr0.6_5/best_bbox_mAP_epoch_22.pth',
                        help='Checkpoint file')
    parser.add_argument('--out-file',
                        default='/home/duomeitinrfx/users/pengxl/mmdetection/tools/work_dirs_0.8_0.2/1000_1300_dcn/prio_gcn_ca/tood/tood_lr0.6_5/2756.jpg',
                        help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
