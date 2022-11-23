# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/dataset/'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#-- image augmentation (TRAIN)
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
#-- image augmentation (TEST)
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

data = dict(
    samples_per_gpu=2,       #-- gpu 하나당 batch size
    workers_per_gpu=2,      #-- pytorch에서 num_worker와 같은 개념
    train=dict(
        type=dataset_type,      #-- CocoDataset
        classes = classes, 
        #-- train annotation file path
        ann_file=data_root + 'train-kfold-0.json',
        #-- img path
        img_prefix=data_root,
        #-- data augmentation
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val-kfold-0.json',
        img_prefix=data_root,
        classes = classes, 
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes = classes, 
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
