_base_ = [
    './_base_/datasets/coco_detection.py', 
    './_base_/models/cascade_rcnn_r50_fpn.py', 
    './_base_/schedules/schedule_1x_CosineAnnealing.py', 
    './_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

log_config = dict(
    interval=50,
    hooks = [
        dict(type='TextLoggerHook'),       
        dict(type='MMDetWandbHook',
            #  wandb init 설정
            init_kwargs=dict(
                entity = 'miho',
                project = 'Detection-Competition', 
                name = '1124_yr_cascade-rcnn_dcn', 
                tags = ['cascade-rcnn', 'dcn', 'adamw', 'cosineannealing', 'resnet50'],
                notes = 'cascade-rcnn의 resnet50 백본 그대로 dcn만 적용해서 학습시킨 것', 
            ),
            interval = 50,
            log_checkpoint=True,   
            log_checkpoint_metadata=True,
            num_eval_images=100,           
            bbox_score_thr=0.3)     
    ])