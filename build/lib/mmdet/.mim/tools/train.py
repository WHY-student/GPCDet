# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)

"""
resume-from 既加载了模型的权重和优化器的状态，也会继承指定 checkpoint 的迭代次数，不会重新开始训练。
load-from 则是只加载模型的权重，它的训练是从头开始的，经常被用于微调模型。
"""
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default='../configs/_myimprovemodel/new_improve/tood_gs.py',help='train config file path')
    # 在训练期间，日志文件和 checkpoint 文件将会被保存在工作目录下，它需要通过配置文件中的 work_dir 或者 CLI 参数中的 --work-dir 来指定。
    #parser.add_argument('--work-dir',default='work_dirs_tood_panfpn/gsconv/gs_w_(1-w)_0.5_3',help='the dir to save logs and models')
    parser.add_argument('--work-dir',default='work_dirs_0.8_0.2/fpn/gs_fusion/gs_fusion0.1',help='the dir to save logs and models')


    #parser.add_argument('--config', default='../configs/_myimprovemodel/new_improve/tood_panfpn.py',help='train config file path') 	
    #parser.add_argument('--work-dir',default='work_dirs_tood_panfpn/panfpnaug/aug_ALTM_trainaug',help='the dir to save logs and models')
    # --resume-from  从某个 checkpoint 文件继续训练.
    parser.add_argument(
        '--resume-from',help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint automatically')
    # --no-validate (不建议): 在训练期间关闭测试.
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=813804253, help='random seed')
    parser.add_argument(

        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')

    # --options 'Key=value': 覆盖使用的配置文件中的其他设置.
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')

    #--cfg - options 可以用来修改配置文件
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    # 学习率自动缩放
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()


    # cfg 包含三个字典信息 filename配置文件的路径，就是超参数中的config
    # text filename就是filename配置文件的配置信息
    #  pretty_test 对上面的信息进行过滤，filename中的信息对base中的会进行继承和重写
    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn('Can not find "auto_scale_lr" or '
                          '"auto_scale_lr.enable" or '
                          '"auto_scale_lr.base_batch_size" in your'
                          ' configuration file. Please update all the '
                          'configuration files to mmdet >= 2.24.1.')

    # 多进程设置
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    """
    这里最终分配到gpu_ids这个列表中，而且只有一个，如果直接运行train
    """
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # 将配置文件存到work_dir目录下
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # 生成log文件
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # 初始化meta为dict以记录一些重要信息，如环境信息和种子，这些信息将被记录
    # 初始化为空
    meta = dict()
    # log 运行环境的信息，例如python版本，cuda信息，gpu信息等等
    env_info_dict = collect_env()
    # 将信息输出到控制台
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)

    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # 输出是否分布式训练
    logger.info(f'Distributed training: {distributed}')
    #输出配置文件的信息
    logger.info(f'Config:\n{cfg.pretty_text}')

    # 当cuda可用时，分配cuda
    cfg.device = get_device()
    # 设置随机数种子，每次训练的时候都能保证，每个batch的图片都能一样
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)
    # 构建模型

    """
    模型的构建取决于cfg.model,cfg.train_cfg,cfg.test_cfg 三个字典
    初始化模型, 函数内部调用的是DETECTORS.build(cfg)
    """
    print("*****************************model*************************************************")
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    """模型参数初始化"""
    model.init_weights()
    # 初始化训练集和验证集, 函数内部调用build_from_cfg(cfg, DATASETS), 等价于DATASETS.build(cfg)
    print("*****************************traindata*************************************************")
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        assert 'val' in [mode for (mode, _) in cfg.workflow]
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.get(  #  验证集在训练过程中使用train pipeline而不是test pipeline
            'pipeline', cfg.data.train.dataset.get('pipeline'))
        print("*****************************valdata*************************************************")
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    print("*****************************传入模型*************************************************")
    # 传入模型和数据集, 准备开始训练模型
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
