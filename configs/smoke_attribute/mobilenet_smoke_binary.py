_base_ = [
    '../_base_/models/mobilenet_v2_1x.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    # '../_base_/schedules/imagenet_bs256_epochstep.py',
    '../_base_/default_runtime.py'
]


model = dict(
    head=dict(
        num_classes=2,
    )
)

# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data_root = '/data1/sunjun/dataset/safemonitor/xiyan/old_version/classification/'

data = dict(
    samples_per_gpu=32*4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix=data_root + 'train',
        ann_file=data_root + 'train/txts/train_all.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=data_root + 'val',
        ann_file=data_root + 'val/txts/val_all.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=data_root + 'val',
        ann_file=data_root + 'val/txts/val_all.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')

# optimizer
optimizer = dict(type='SGD', lr=0.045, momentum=0.9, weight_decay=0.00004)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=100)
load_from = '/home/sunjun/code/github/mmclassification/pretrain/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
