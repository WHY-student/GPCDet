# optimizer
optimizer = dict(type='SGD',# 用于构建优化器的配置文件
                 lr=0.004, momentum=0.9, weight_decay=0.0001)# SGD 的衰减权重
optimizer_config = dict(grad_clip=None) # optimizer hook 的配置文件
# learning policy
lr_config = dict(  # 学习率调整配置，用于注册 LrUpdater hook
    policy='step',   # 调度流程(scheduler)的策略
    warmup='linear', # warmup策略
    warmup_iters=500,  # warmup的迭代次数
    warmup_ratio=0.001, # 用于warmup的起始学习率的比率
    step=[10,16]) # 衰减学习率的起止回合数
runner = dict(type='EpochBasedRunner', max_epochs=25)
