_base_ = [
    '/opt/ml/teajun/level2_objectdetection_cv-level2-cv-11/baseline/UniverseNet/configs/_focalnet_/_base_/models/atss_focalnet_fpn.py',
    '/opt/ml/teajun/level2_objectdetection_cv-level2-cv-11/baseline/UniverseNet/configs/_focalnet_/_base_/datasets/coco_detection_detraug.py',
    './_base_/schedules/schedule_1x.py', './_base_/default_runtime.py'
]

pretrained = 'https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_tiny_srf.pth'  # noqa
model = dict(
    backbone=dict(
        type='FocalNet',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        focal_windows=[11, 9, 9, 7],
        focal_levels=[2, 2, 2, 2],
        use_conv_embed=False,
        pretrained=pretrained),
    neck=dict(in_channels=[192, 384, 768]))

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
lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)

fp16 = dict(loss_scale=dict(init_scale=512))
