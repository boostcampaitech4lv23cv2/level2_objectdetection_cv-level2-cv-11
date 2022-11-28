_base_ = [
    '../_base_/datasets/coco_detection.py', 
    '../_base_/models/faster_rcnn_r50_fpn.py', 
    '../_base_/schedules/schedule_1x_CosineAnnealing.py', 
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
    neck=dict(in_channels=[96, 192, 384, 768]))

optimizer = dict(
    _delete_=True, 
    type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    grad_clip= dict(max_norm=35, norm_type=2)
)     

log_config = dict(
    hooks = [
        dict(type='TextLoggerHook'),       
        dict(type='MMDetWandbHook',
             
            #  wandb init 설정
            init_kwargs=dict(
                entity = 'miho',
                project = 'Detection-Competition', 
                name = '1128_yr_faster-rcnn_swin_csa_step_sgd', 
                tags = ['swin', 'sgd', 'faster_rcnn', 'cosineannealing', 'ce'],
                notes = 'faster-rcnn, swin의 다양한 설정 실험', 
            ),
            # Logging interval (iterations)
            interval = 50,
            log_checkpoint=False,
            log_checkpoint_metadata=True,
            num_eval_images=100,           
            bbox_score_thr=0.3)     
    ]
)


evaluation = dict(
    interval=1, 
    metric=['bbox'], 
    save_best = 'bbox_mAP_50'
)
