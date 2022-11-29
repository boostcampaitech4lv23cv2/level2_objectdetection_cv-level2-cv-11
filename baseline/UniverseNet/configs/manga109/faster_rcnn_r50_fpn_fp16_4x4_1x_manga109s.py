_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/manga109s.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(roi_head=dict(bbox_head=dict(num_classes=4)))

data = dict(samples_per_gpu=4)

optimizer = dict(type='SGD', lr=0.16, momentum=0.9, weight_decay=0.0001)
lr_config = dict(warmup_iters=500)

evaluation = dict(classwise=True)

fp16 = dict(loss_scale=512.)

load_from = 'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa