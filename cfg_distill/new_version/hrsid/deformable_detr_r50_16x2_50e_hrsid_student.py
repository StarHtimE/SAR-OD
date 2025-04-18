_base_ = [
    '../../../configs/_base_/datasets/HRSID.py', '../../../configs/_base_/default_runtime.py'
]
model = dict(
    type='DistillDeformableDETR',
    num_queries=300,
    num_feature_levels=4,
    with_box_refine=False,
    as_two_stage=False,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),

    encoder=dict(  # DeformableDetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1))),
    decoder=dict(  # DeformableDetrTransformerDecoder
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(  # DeformableDetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),

    bbox_head=dict(
        type='DistillDeformableDETRHead',
        num_classes=1,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        loss_cls_distill=dict(
            type='DistillCrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_bbox_distill=dict(type='L1Loss', loss_weight=5.0),
        loss_iou_distill=dict(type='GIoULoss', loss_weight=2.0)),

    # bbox_head=dict(
    #     type='DistillDeformableDETRHead',
    #     num_query=300,
    #     num_classes=80,
    #     in_channels=2048,
    #     sync_cls_avg_factor=True,
    #     as_two_stage=False,
    #     transformer=dict(
    #         type='DeformableDetrTransformer',
    #         encoder=dict(
    #             type='DetrTransformerEncoder',
    #             num_layers=6,
    #             transformerlayers=dict(
    #                 type='BaseTransformerLayer',
    #                 attn_cfgs=dict(
    #                     type='MultiScaleDeformableAttention', embed_dims=256),
    #                 feedforward_channels=1024,
    #                 ffn_dropout=0.1,
    #                 operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
    #         decoder=dict(
    #             type='DeformableDetrTransformerDecoder',
    #             num_layers=6,
    #             return_intermediate=True,
    #             transformerlayers=dict(
    #                 type='DetrTransformerDecoderLayer',
    #                 attn_cfgs=[
    #                     dict(
    #                         type='MultiheadAttention',
    #                         embed_dims=256,
    #                         num_heads=8,
    #                         dropout=0.1),
    #                     dict(
    #                         type='MultiScaleDeformableAttention',
    #                         embed_dims=256)
    #                 ],
    #                 feedforward_channels=1024,
    #                 ffn_dropout=0.1,
    #                 operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
    #                                  'ffn', 'norm')))),
    #     positional_encoding=dict(
    #         type='SinePositionalEncoding',
    #         num_feats=128,
    #         normalize=True,
    #         offset=-0.5),
    #     loss_cls=dict(
    #         type='FocalLoss',
    #         use_sigmoid=True,
    #         gamma=2.0,
    #         alpha=0.25,
    #         loss_weight=2.0),
    #     loss_bbox=dict(type='L1Loss', loss_weight=5.0),
    #     loss_iou=dict(type='GIoULoss', loss_weight=2.0),
    #     loss_cls_distill=dict(
    #         type='DistillCrossEntropyLoss',
    #         use_sigmoid=True,
    #         loss_weight=1.0),
    #     loss_bbox_distill=dict(type='L1Loss', loss_weight=5.0),
    #     loss_iou_distill=dict(type='GIoULoss', loss_weight=2.0)),
    
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)),
        distill_assigner=dict(
            type='DistillHungarianAssigner',
            cls_cost=dict(type='DistillCrossEntropyLossCost', weight=1.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)),
            ),
    test_cfg=dict(max_per_img=100))

# # train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# # from the default setting in mmdet.
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(
#         type='AutoAugment',
#         policies=[
#             [
#                 dict(
#                     type='Resize',
#                     img_scale=[(480, 1333), (512, 1333), (544, 1333),
#                                (576, 1333), (608, 1333), (640, 1333),
#                                (672, 1333), (704, 1333), (736, 1333),
#                                (768, 1333), (800, 1333)],
#                     multiscale_mode='value',
#                     keep_ratio=True)
#             ],
#             [
#                 dict(
#                     type='Resize',
#                     # The radio of all image in train dataset < 7
#                     # follow the original impl
#                     img_scale=[(400, 4200), (500, 4200), (600, 4200)],
#                     multiscale_mode='value',
#                     keep_ratio=True),
#                 dict(
#                     type='RandomCrop',
#                     crop_type='absolute_range',
#                     crop_size=(384, 600),
#                     allow_negative_crop=True),
#                 dict(
#                     type='Resize',
#                     img_scale=[(480, 1333), (512, 1333), (544, 1333),
#                                (576, 1333), (608, 1333), (640, 1333),
#                                (672, 1333), (704, 1333), (736, 1333),
#                                (768, 1333), (800, 1333)],
#                     multiscale_mode='value',
#                     override=True,
#                     keep_ratio=True)
#             ]
#         ]),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=1),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]
# # test_pipeline, NOTE the Pad's size_divisor is different from the default
# # setting (size_divisor=32). While there is little effect on the performance
# # whether we use the default setting or use size_divisor=1.
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=1),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img'])
#         ])
# ]
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(filter_empty_gt=False, pipeline=train_pipeline),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))

## optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

# learning policy
max_epochs = 50
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=32)
