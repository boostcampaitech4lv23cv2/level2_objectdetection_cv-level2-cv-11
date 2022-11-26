# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,     # 10 -> 우리 데이터셋에 맞게
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        # anchor 생성시 설정 - 어떤 dataset의 문제냐에 따라 초기값이 달라질 수 있음
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],     # width/height 비율
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        # classification loss 어떤 것 사용할지
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        # bbox loss를 어떤 것 사용할지 
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,    # possitive 판단할 때 기준이 되는 IoU Threshold
            neg_iou_thr=0.4,    # negative 판단할 때 기준이 되는 IoU Threshold
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,   
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),    # mAP50으로 측정할 때는 iou_thresh 0.5 좋음. (map70, 90등 올라가면 iou_thresh도 올려주는 것이 좋음)
        max_per_img=100))   # 이미지 하나당 추출하는 object 최대 개수, 데이터에 따라
