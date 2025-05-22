find_unused_parameters = True
num_fiducial = 8
tps_size = (0.25, 1)
model = dict(
    type='LSG',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=2, fallback_on_stride=False),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        stage_with_dcn=(False, True, True, True)
    ),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True,
        act_cfg=None),
    bbox_head=dict(
        type='mmdet.LSGDetHead',
        in_channels=256,
        loss=dict(type='mmdet.LSGDetLoss')),
    recog_head=dict(
        type='LocalGridHead',
        embed_dims=256,
        num_feature_levels=4,
        num_reg_fcs=1,
        with_poly=False,
        decoder=dict(
            grid_size=(5, 5),
            n_level=4,
            n_layers=6,
            d_embedding=256,
            n_head=8,
            d_k=32,
            d_v=32,
            d_model=256,
            d_inner=1024,
            n_position=200,
            dropout=0.1,
            disturb_range=1.0,
            disturb_vert=True,
            add_pos=True,
            boxfix=False,
            # sample_mode = "nearest",
        ),
        label_convertor=dict(type='AttnConvertor', lower=True, max_seq_len=25,
                             # dict_file='./mmocr//chn_char_list.txt'
                             ),
        reg_loss=dict(type='TFLoss', reduction='mean')
    )
)

train_cfg = None
# test_cfg = dict(
#     use_gt=True
# )
test_cfg = None
dataset_type = 'IcdarE2EDataset'
data_root = '/data_center/scene-text/synthtext-150k/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile',
         # to_float32=True
         ),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.2,
        contrast=0.2
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomScaling', size=800, scale=(3. / 4, 5. / 2)),
    dict(
        type='RandomCropPolyInstancesWithText',
        instance_key='gt_masks',
        crop_ratio=0.6,
        min_side_ratio=0.1),
    dict(
        type='RandomRotatePolyInstances',
        rotate_ratio=0.5,
        max_angle=90,
        pad_with_fixed_color=False),
    dict(type='SquareResizePad', target_size=800, pad_ratio=0.6),  # 448 not 640
    # dict(type='ResizePad', target_size=(640,1024), pad_ratio=0.5),
    dict(type='RandomFlip', flip_ratio=0., direction='horizontal'),
    dict(type='Pad', size_divisor=32),
    # dict(type='Pad', size=(640,1080)),
    dict(
        type='LSGTargets',
    ),
    dict(
        type='CustomFormatBundle',
        keys=['gt_texts', 'gt_reference_points'],
        visualize=dict(flag=False, boundary_key=None)),
    dict(type='Collect',
         keys=['img', 'gt_texts',
               'gt_reference_points', 'gt_text_mask',
               'gt_head_mask', 'gt_center_mask',
               ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1500, 960)],
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1080, 720), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
    # dict(
    #     type='MultiScaleFlipAug',
    #     img_scale=(1500, 960),
    #     flip=False,
    #     transforms=[
    #         dict(type='Resize', img_scale=(1080, 720), keep_ratio=True),
    #         dict(type='Normalize', **img_norm_cfg),
    #         dict(type='Pad', size_divisor=32),
    #         dict(type='RandomFlip', flip_ratio=0., direction='horizontal'),
    #             dict(
    #                 type='LSGTargets',
    #             ),
    #         dict(
    #             type='CustomFormatBundle',
    #             keys=['gt_texts', 'gt_reference_points'],
    #             visualize=dict(flag=False, boundary_key=None)),
    #         dict(
    #             type='Collect',
    #             keys=['img', 'gt_texts',
    #                   'gt_reference_points'
    #                   ]
    #         )
    #     ]

    # )
]
_base_ = [
    '../_base_/det_datasets/pretrain_data.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline)
)
evaluation = dict(interval=10000000, metric='eval_hmean', by_epoch=False)

# optimizer
# optimizer = dict(type='SGD', lr=1e-3, momentum=0.90, weight_decay=5e-4)
optimizer = dict(type='AdamW', lr=1e-4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step', step=[300000, 450000], by_epoch=False,
    warmup='linear', warmup_iters=5000, warmup_ratio=0.001,
)
runner = {
    'type': 'IterBasedRunner',
    'max_iters': 600000
}

checkpoint_config = dict(interval=10000, by_epoch=False)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook')
    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
