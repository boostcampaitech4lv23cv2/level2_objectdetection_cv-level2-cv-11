dataset=dict(
    type='CocoDataset',
    ann_file='/opt/ml/dataset/train-kfold-0.json',
    img_prefix='/opt/ml/dataset/',
    classes=('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
              'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True)
    ],
    filter_empty_gt=False
)