_base_ = [
    '../_base_/datasets/coco_detection.py', 
    '../_base_/models/cascade_rcnn_r50_fpn_ce.py', 
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
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
    neck=dict(in_channels=[96, 192, 384, 768])
)

log_config = dict(
    interval=50,
    hooks = [
        dict(type='TextLoggerHook'),       
        dict(type='MMDetWandbHook',
             
            #  wandb init 설정
            init_kwargs=dict(
                entity = 'miho',
                project = 'Detection-Competition', 
                name = '1128_yr_cascade-rcnn_swin_ce_step_sgd', 
                tags = ['cascade-rcnn', 'swin', 'sgd', 'ce', 'steplr'],
                notes = 'cascade-rcnn에 swin으로 backbone 변경하고, cross entropy로 학습, steplr,sgd 사용', 
            ),
            # Logging interval (iterations)
            interval = 50,
            log_checkpoint=True,   
            log_checkpoint_metadata=True,
            num_eval_images=100,           
            bbox_score_thr=0.3)     
])