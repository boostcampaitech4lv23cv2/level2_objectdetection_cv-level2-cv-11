checkpoint_config = dict(
    interval = 1      # 모델이 저장되는 epoch 주기
)

# yapf:disable
log_config = dict(
    interval=50,
    hooks = [
        dict(type='TextLoggerHook'),       
        dict(type='MMDetWandbHook',
             
            #  wandb init 설정
            init_kwargs=dict(
                entity = 'miho',
                project = 'Detection-Competition', 
                name = '실험일_수행인_model_특이사항', 
                tags = ['최대한 많이, backbone, loss, dataset 등등 '],
                notes = 'cascade-rcnn', 
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
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)