_base_ = [
    './_base_/datasets/coco_detection.py', 
    './_base_/models/faster_rcnn_r50_fpn.py', 
    './_base_/schedules/schedule_1x_CosineAnnealing.py', 
    './_base_/default_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
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
    neck=dict(in_channels=[96, 192, 384, 768]), 
    roi_head=dict(
        type='DynamicRoIHead',
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn_proposal=dict(nms=dict(iou_threshold=0.85)),
        rcnn=dict(
            dynamic_rcnn=dict(
                iou_topk=75,
                beta_topk=10,
                update_iter_interval=100,
                initial_iou=0.4,
                initial_beta=1.0))),
    test_cfg=dict(rpn=dict(nms=dict(iou_threshold=0.85)))
)


log_config = dict(
    hooks = [
        dict(type='TextLoggerHook'),       
        dict(type='MMDetWandbHook',
            #  wandb init 설정
            init_kwargs=dict(
                entity = 'miho',
                project = 'Detection-Competition', 
                name = '1129_yr_dynamic-rcnn_swin_ce_csa_adamw', 
                tags = ['swin', 'adamw', 'dynamic_rcnn', 'cosineannealing', 'ce'],
                notes = 'dynamic-rcnn, swin의 실험', 
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
