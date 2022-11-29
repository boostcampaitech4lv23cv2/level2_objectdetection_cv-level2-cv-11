_base_ = [
    '../universenet/models/universenet101_2008d.py',
    '../_base_/datasets/trash_mstrain_512_1024_tta1.py', # Trash Dataset
    '../_base_/schedules/schedule_20e.py', '../_base_/default_runtime.py'
]

model = dict(
    # SyncBN is used in universenet50_2008.py
    # If total batch size < 16, please change BN settings of backbone.
    backbone=dict(
        norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True),
    # iBN of SEPC is used in universenet50_2008.py
    # If samples_per_gpu < 4, please change BN settings of SEPC.
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
        dict(
            type='SEPC',
            out_channels=256,
            stacked_convs=4,
            pconv_deform=False,
            lcconv_deform=True,
            ibn=True,
            pnorm_eval=True,  # please set True if samples_per_gpu < 4
            lcnorm_eval=True,  # please set True if samples_per_gpu < 4
            lcconv_padding=1)
    ],
    bbox_head=dict(num_classes=10))  # please change for your dataset

data = dict(samples_per_gpu=2)

evaluation = dict(
    interval = 1, # validation 주기
    metric = ['bbox'],
    classwise = True,
    save_best = 'bbox_mAP_50'
)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=1000)

fp16 = dict(loss_scale=512.)

runner = dict(
    type = 'EpochBasedRunner',
    max_epochs = 30
)

log_config = dict(
    hooks = [
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
            init_kwargs=dict(
                entity = 'miho',
                project = 'Detection-Competition',
                name = '1128_kh_universenet101_2008d-tta1', 
                tags = ['universenet', 'sgd', 'res2net101', 'mstrain_512_1024', 'tta'],
                notes = 'UniverseNet101_2008d TTA 실험', 
            ),
            interval = 50,
        )     
    ]
)

load_from = 'https://github.com/shinya7y/UniverseNet/releases/download/20.10/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco_20201023_epoch_20-3e0d236a.pth'