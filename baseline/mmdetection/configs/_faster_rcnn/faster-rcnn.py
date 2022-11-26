_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
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
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train-kfold-0.json',
        img_prefix=data_root,
        pipeline=train_pipeline,
        classes=classes
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val-kfold-0.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes
        ))
evaluation = dict(interval=1, metric='bbox')

seed = 2022

model = dict(
  roi_head = dict(
    bbox_head = dict(
      num_classes = 10
    )
  )
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
             
            #  wandb init 설정
            init_kwargs=dict(
                entity = 'miho',
                project = 'Detection-Competition', 
                name = '1124_kh_faster-rcnn_baseline', 
                tags = ['resnet50', 'sgd', 'faster_rcnn', '512', 'ce'],
                notes = '베이스라인 Faster RCNN 성능 측정용', 
            ),
            
            # Logging interval (iterations)
            interval = 50,
            
            # Save the checkpoint at every checkpoint interval as W&B Artifacts. Default False
            # You can reliably store these checkpoints as W&B Artifacts by using the log_checkpoint=True argument in MMDetWandbHook. 
            # This feature depends on the MMCV's CheckpointHook that periodically save the model checkpoints. 
            # The period is determined by checkpoint_config.interval.
            log_checkpoint=False,
            
                  
            log_checkpoint_metadata=True,
            
            # The number of validation images to be logged. If zero, the evaluation won't be logged. Defaults to 100.
            num_eval_images=100,           
            
            # Threshold for bounding box scores. Defaults to 0.3.
            bbox_score_thr=0.3)     
    ])

runner = dict(
    type = 'EpochBasedRunner',
    max_epochs = 20
)