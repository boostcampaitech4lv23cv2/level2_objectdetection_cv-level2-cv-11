_base_ = [
    '../_base_/datasets/coco_detection.py', 
    '../_base_/models/cascade_rcnn_r50_fpn_ce.py', 
    '../_base_/schedules/schedule_1x_CosineAnnealing.py', 
    '../_base_/default_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
model = dict(
    backbone=dict(
        _delete_=True,      # 덮어쓰기
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],   
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        in_channels=[384, 768, 1536],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5)
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#-- image augmentation (TRAIN)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
#-- image augmentation (TEST)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

lr_config = dict(
    policy='CosineRestart',
    periods=[8, 8, 8, 8, 8, 8],
    restart_weights=[1, 0.7, 0.5, 0.2, 0.01, 0.01]
)
runner = dict(max_epochs=48)

log_config = dict(
    interval=50,
    hooks = [
        dict(type='TextLoggerHook'),       
        dict(type='MMDetWandbHook',
             
            #  wandb init 설정
            init_kwargs=dict(
                entity = 'miho',
                project = 'Detection-Competition', 
                name = '1130_yr_cascade-rcnn_swinL_ce_csr2_adamw', 
                tags = ['cascade-rcnn', 'swin-large', 'adamw', 'ce', 'cosinerestart'],
                notes = 'cascade-rcnn에 swin의 다양한 실험', 
            ),
            # Logging interval (iterations)
            interval = 50,
            log_checkpoint=False,   
            log_checkpoint_metadata=True,
            num_eval_images=100,           
            bbox_score_thr=0.3)     
])

evaluation = dict(
    interval=1, 
    metric=['bbox'], 
    save_best = 'bbox_mAP_50'
)