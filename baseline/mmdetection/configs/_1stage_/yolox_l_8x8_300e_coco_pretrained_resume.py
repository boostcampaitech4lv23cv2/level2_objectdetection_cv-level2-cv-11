_base_ = './yolox_s_8x8_300e_coco.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256))

resume_from='/opt/ml/teajun/level2_objectdetection_cv-level2-cv-11/baseline/mmdetection/work_dirs/yolox_l_8x8_300e_coco_pretrained/latest.pth'