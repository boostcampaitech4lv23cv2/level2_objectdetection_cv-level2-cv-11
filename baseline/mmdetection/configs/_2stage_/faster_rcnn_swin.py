_base_ = [
    './_base_/datasets/coco_detection.py', 
    './_base_/models/faster_rcnn_r50_fpn.py', 
    './_base_/schedules/schedule_1x.py', 
    './_base_/default_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,      # 덮어쓰기
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=1000, step=[8, 11])
runner = dict(max_epochs=12)

# log_config - 다양한 로거 후크를 초기화
log_config = dict(
    hooks = [
        dict(type='TextLoggerHook'),       
        dict(type='MMDetWandbHook',
             
            #  wandb init 설정
            init_kwargs=dict(
                entity = 'miho',
                project = 'Detection-Competition', 
                name = '1123_yr_faster-rcnn', 
                tags = ['swin', 'adamw', 'faster_rcnn'],
                notes = 'faster-rcnn, swin으로 학습 및 wandb 설정 테스트해본 것', 
            ),
            
            # Logging interval (iterations)
            interval = 50,
            
            # Save the checkpoint at every checkpoint interval as W&B Artifacts. Default False
            # You can reliably store these checkpoints as W&B Artifacts by using the log_checkpoint=True argument in MMDetWandbHook. 
            # This feature depends on the MMCV's CheckpointHook that periodically save the model checkpoints. 
            # The period is determined by checkpoint_config.interval.
            log_checkpoint=True,
            
                  
            log_checkpoint_metadata=True,
            
            # The number of validation images to be logged. If zero, the evaluation won't be logged. Defaults to 100.
            num_eval_images=100,           
            
            # Threshold for bounding box scores. Defaults to 0.3.
            bbox_score_thr=0.3)     
    ]
)

# checkpoint_config - MMCV의 CheckpointHook을 초기화
# 구성 섹션에서 정의한 간격으로 모델 체크포인트를 저장
checkpoint_config = dict(
    interval = 4
)

runner = dict(
    type = 'EpochBasedRunner',
    max_epochs = 20
)