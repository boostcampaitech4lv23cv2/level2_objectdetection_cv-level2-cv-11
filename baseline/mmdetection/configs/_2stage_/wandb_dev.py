_base_ = [
    './_base_/datasets/coco_detection.py', 
    './_base_/models/faster_rcnn_r50_fpn.py', 
    './_base_/schedules/schedule_1x.py', 
    './_base_/default_runtime.py'
]

data = dict(
    train=dict(ann_file='/opt/ml/dataset/val-tiny.json'),
    val=dict(ann_file = '/opt/ml/dataset/val-tiny.json')
)

# log_config - 다양한 로거 후크를 초기화
log_config = dict(
    hooks = [
        dict(type='TextLoggerHook'),       
        dict(type='MMDetWandbHook',
             
            #  wandb init 설정
            init_kwargs=dict(
                entity = 'miho',
                project = 'dev', 
                name = 'wandb-dev-2', 
                tags = ['dev'],
                notes = 'wandb 개발 테스트용', 
            ),
            
            # Logging interval (iterations)
            interval = 50,
            
            # Save the checkpoint at every checkpoint interval as W&B Artifacts. Default False
            # You can reliably store these checkpoints as W&B Artifacts by using the log_checkpoint=True argument in MMDetWandbHook. 
            # This feature depends on the MMCV's CheckpointHook that periodically save the model checkpoints. 
            # The period is determined by checkpoint_config.interval.
            log_checkpoint=True,
            
                  
            log_checkpoint_metadata=True,
            
            # The number of validation images to be logged. If zero, the evaluation won't be logged. Defaults to 100.
            num_eval_images=1,
            
            # Threshold for bounding box scores. Defaults to 0.3.
            bbox_score_thr=0.3,

            )     
    ]
)

# checkpoint_config - MMCV의 CheckpointHook을 초기화
# 구성 섹션에서 정의한 간격으로 모델 체크포인트를 저장
checkpoint_config = dict(
    interval = 4
)

runner = dict(
    type = 'EpochBasedRunner',
    max_epochs = 20
)

evaluation = dict(
    interval = 1,
    metric = ['bbox'],
    classwise = True,
    save_best = 'bbox_mAP_50'
)