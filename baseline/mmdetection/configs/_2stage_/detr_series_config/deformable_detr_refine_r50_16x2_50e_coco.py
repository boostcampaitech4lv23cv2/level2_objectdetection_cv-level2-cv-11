_base_ = 'deformable_detr_r50_16x2_50e_coco.py'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.pth'
model = dict(bbox_head=dict(with_box_refine=True))
