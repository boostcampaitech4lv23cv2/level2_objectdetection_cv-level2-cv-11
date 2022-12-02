# dataset settings
data_root = '/opt/ml/dataset/'
dataset_type = 'CocoDataset'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# augmentation strategy originates from DETR, except for size_divisor=32

# albu_train_transforms = [
#     # dict(
#     #     type='ShiftScaleRotate',
#     #     shift_limit=0.0625,
#     #     scale_limit=0.0,
#     #     rotate_limit=0,
#     #     interpolation=1,
#     #     p=0.5),
#     # dict(
#     #     type='RandomBrightnessContrast',
#     #     brightness_limit=[0.1, 0.3],
#     #     contrast_limit=[0.1, 0.3],
#     #     p=0.2),
#     dict(
#         type='OneOf',
#         transforms=[
#             # dict(
#             #     type='RGBShift',
#             #     r_shift_limit=10,
#             #     g_shift_limit=10,
#             #     b_shift_limit=10,
#             #     p=1.0),
#             dict(
#                 type='HueSaturationValue',
#                 hue_shift_limit=20,
#                 sat_shift_limit=30,
#                 val_shift_limit=20,
#                 p=1.0)
#         ],
#         p=0.1),
#     dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
#     # dict(type='ChannelShuffle', p=0.1),
#     # dict(
#     #     type='OneOf',
#     #     transforms=[
#     #         dict(type='Blur', blur_limit=3, p=1.0),
#     #         dict(type='MedianBlur', blur_limit=3, p=1.0)
#     #     ],
#     #     p=0.1),
# ]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                     (576, 1333), (608, 1333), (640, 1333),
                                     (672, 1333), (704, 1333), (736, 1333),
                                     (768, 1333), (800, 1333)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes = classes, 
        ann_file=data_root + 'train-kfold-0-pseudo-3.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes = classes, 
        ann_file=data_root + 'val-kfold-0.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes = classes, 
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
