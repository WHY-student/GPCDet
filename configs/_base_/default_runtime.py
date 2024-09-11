# 型保存间隔
checkpoint_config = dict(interval=16)
# yapf:disable
log_config = dict(
    interval=50, # 打印日志的间隔
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')  # 用于设置分布式训练的参数，端口也同样可被设置
log_level = 'INFO'  # 日志的级别
load_from = None  # 从一个给定路径里加载模型作为预训练模型
resume_from = None   # 从给定路径里恢复检查点(checkpoints)，训练模式将从检查点保存的轮次开始恢复训练
workflow = [('train', 1)] # runner 的工作流程，[('train', 1)] 表示只有一个工作流且工作流仅执行一次

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
