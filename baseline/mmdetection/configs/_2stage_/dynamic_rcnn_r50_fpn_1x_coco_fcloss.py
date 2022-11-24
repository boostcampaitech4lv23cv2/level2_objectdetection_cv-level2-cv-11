_base_ = [
    './_base_/models/faster_rcnn_r50_fpn_foloss.py',
    './_base_/datasets/coco_detection.py',
    './_base_/schedules/schedule_1x.py', './_base_/default_runtime.py'
]
model = dict(
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
    test_cfg=dict(rpn=dict(nms=dict(iou_threshold=0.85))))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
            #  wandb init 설정
            init_kwargs=dict(
                entity = 'miho',
                project = 'Detection-Competition', 
                name = '1124_gh_dynamic-rcnn_focalloss', 
                tags = ['dynamic_rcnn', 'resnet50', 'focal', 'smoothl1', 'coco', 'sgd', 'steplr', 'warmup'],
                notes = 'dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py 에서 faster_rcnn_r50_fpn_foloss만 바꾼 것', 
            ),
             
            # Logging interval (iterations)
            interval = 10,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            # The number of validation images to be logged. If zero, the evaluation won't be logged. Defaults to 100.
            num_eval_images=100,
            # Threshold for bounding box scores. Defaults to 0.3.
            bbox_score_thr=0.3
        )
    ]
)