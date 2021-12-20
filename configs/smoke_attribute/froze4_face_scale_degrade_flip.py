_base_ = [
    '../_base_/models/mobilenet_v2_1x.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    # '../_base_/schedules/imagenet_bs256_epochstep.py',
    '../_base_/default_runtime.py'
]

file_type=[dict(type='ce', max_len=2), dict(type='bce', max_len=2, class_name=['Smoke', 'InsulatingGlove'], class_wise=False)]

model = dict(
    backbone=dict(frozen_stages=2),
    head=dict(
        type='HiearachicalLinearClsHead',
        file_type=file_type,
        num_classes=4,
        loss=dict(type='HierarchicalCrossEntropyLoss', use_focal=True, alpha=0.45, gamma=1.5, loss_weight=1.0, split=(2, 4), use_sigmoid=(False, True), use_soft=(False, False),),
    )
)

# dataset settings
dataset_type = 'HierarchicalDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline1 = [
    dict(type='LoadImageFromFile'),
    dict(type='Rotate', angle=180.0, auto_bound=True),
    dict(type='Resize', size=(256, 256), backend='pillow'),
    dict(type='Albu',
         transforms=[
             # dict(type='PadIfNeeded', min_height=276, min_width=276, p=0.5),
             # dict(type='RandomCrop', height=256, width=256, p=1.0),
             # dict(
             #     type='ShiftScaleRotate',
             #     shift_limit=0.0625,
             #     scale_limit=0.0,
             #     rotate_limit=0,
             #     interpolation=1,
             #     p=0.5),
             dict(
                 type='RandomBrightnessContrast',
                 brightness_limit=[0.1, 0.3],
                 contrast_limit=[0.1, 0.3],
                 p=0.5),
             # dict(type='ChannelShuffle', p=0.1),
             # dict(type='RandomScale', scale_limit=(0.2, 0.5)),
             # dict(type='Resize', height=64, width=64, p=1),
             dict(
                 type='OneOf',
                 transforms=[
                     dict(type='JpegCompression', quality_lower=5, quality_upper=20, p=1.0),
                     dict(type='Blur', blur_limit=7, p=1.0),
                     dict(type='MedianBlur', blur_limit=7, p=1.0),
                     dict(type='ISONoise', p=1.0),
                     dict(type='MotionBlur', blur_limit=7, p=1.0),
                 ],
                 p=0.5),
             # dict(
             #     type='OneOf',
             #     transforms=[
             #         dict(type='Blur', blur_limit=3, p=1.0),
             #         dict(type='MedianBlur', blur_limit=3, p=1.0)
             #     ],
             #     p=1.0),
         ],
    ),
    # dict(type='RandomResizedCrop', size=224, scale=(0.8, 1.0), backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
train_pipeline2 = [
    dict(type='LoadImageFromFile'),
    dict(type='Rotate', angle=180.0, auto_bound=True),
    dict(type='Resize', size=(256, 256), backend='pillow'),
    dict(type='Albu',
         transforms=[
             # dict(type='PadIfNeeded', min_height=276, min_width=276, p=0.5),
             # dict(type='RandomCrop', height=256, width=256, p=1.0),
             # dict(
             #     type='ShiftScaleRotate',
             #     shift_limit=0.0625,
             #     scale_limit=0.0,
             #     rotate_limit=0,
             #     interpolation=1,
             #     p=0.5),
             dict(
                 type='RandomBrightnessContrast',
                 brightness_limit=[0.1, 0.3],
                 contrast_limit=[0.1, 0.3],
                 p=0.5),
             # dict(type='ChannelShuffle', p=0.1),
             # dict(type='RandomScale', scale_limit=(0.2, 0.5)),
             # dict(type='Resize', height=64, width=64, p=1),
             # dict(type='JpegCompression', quality_lower=5, quality_upper=20, p=0.5),
             # dict(
             #     type='OneOf',
             #     transforms=[
             #         dict(type='Blur', blur_limit=3, p=1.0),
             #         dict(type='MedianBlur', blur_limit=3, p=1.0)
             #     ],
             #     p=1.0),
         ],
    ),
    # dict(type='RandomResizedCrop', size=224, scale=(0.8, 1.0), backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, 256), backend='pillow'),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data_root = '/data1/sunjun/dataset/safemonitor/xiyan/old_version/classification/'
data_root2 = '/data1/sunjun/dataset/safemonitor/xiyan/huoqiu/classification/'
data = dict(
    samples_per_gpu=32*4,
    workers_per_gpu=4,
    train=[dict(
            type=dataset_type,
            data_prefix=data_root + 'train',
            ann_file=[data_root + 'train/txts/multi-label/train_ml_all_noperson.txt', data_root + 'train/txts/multi-label/train_attr_all_noperson.txt'],
            file_type=file_type,
            pipeline=train_pipeline1
    ),
        dict(
            type=dataset_type,
            data_prefix=data_root + 'new_train/face/ims',
            ann_file=[data_root + 'new_train/face/train_base_attr_face.txt', data_root + 'new_train/face/train_base_attr_face_attr.txt'],
            file_type=file_type,
            pipeline=train_pipeline1
    ),
        dict(
            type=dataset_type,
            data_prefix=data_root + 'train',
            ann_file=[data_root + 'train/txts/multi-label/train_ml_only_hq_withsomesmoke.txt', data_root + 'train/txts/multi-label/train_attr_only_hq_withsomesmoke.txt'],
            file_type=file_type,
            pipeline=train_pipeline2
    ),
        dict(
            type=dataset_type,
            data_prefix='/data1/sunjun/dataset/safemonitor/xiyan/attr_face/attr_test_face/train',
            ann_file=['/data1/sunjun/dataset/safemonitor/xiyan/attr_face/attr_test_face/train_ml.txt', '/data1/sunjun/dataset/safemonitor/xiyan/attr_face/attr_test_face/train_attr.txt'],
            file_type=file_type,
            pipeline=train_pipeline2
    ),
        dict(
            type=dataset_type,
            data_prefix='/home/sunjun/dataset/coco/crop_hand2',
            ann_file=['/home/sunjun/dataset/coco/pfh_train_ml_1w.txt', '/home/sunjun/dataset/coco/pfh_train_attr_1w.txt'],
            file_type=file_type,
            pipeline=train_pipeline2
    ),
        dict(
            type=dataset_type,
            data_prefix='/home/sunjun/dataset/safemonitor/xiyan/v2/imgs',
            ann_file=['/home/sunjun/dataset/safemonitor/xiyan/v2/ml.txt', '/home/sunjun/dataset/safemonitor/xiyan/v2/attr.txt'],
            file_type=file_type,
            pipeline=train_pipeline2
    ),
    ],
    val=dict(
            type=dataset_type,
            data_prefix=data_root + 'val',
            ann_file=[data_root + 'val/txts/multi-label/hq_ml_1207.txt', data_root + 'val/txts/multi-label/hq_attr_1207.txt'],
            file_type=file_type,
            pipeline=test_pipeline
    ),
    # val=dict(
    #     type=dataset_type,
    #     ann_file=[data_root + 'val/txts/multi-label/val_ml_all_hq.txt', data_root + 'val/txts/multi-label/val_attr_all_hq.txt'],
    #     file_type=file_type,
    #     data_prefix=data_root + 'val',
    #     pipeline=test_pipeline
    # ),
    test=dict(
        type=dataset_type,
        ann_file=[data_root + 'val/txts/multi-label/val_ml_all_hq_o_new.txt', data_root + 'val/txts/multi-label/val_attr_all_hq_o_new.txt'],
        file_type=file_type,
        data_prefix=data_root + 'val',
        pipeline=test_pipeline
    ),
)
evaluation = dict(interval=1, metric='accuracy')

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.00004)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='Step', step=[60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)

# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

load_from = '/home/sunjun/code/github/mmclassification/pretrain/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
