_base_ = [
    '../../configs/_base_/models/faster-rcnn_r50_fpn.py', 
    '../../configs/_base_/datasets/SAR_AIRcraft.py',
    '../../configs/_base_/schedules/schedule_1x.py', '../../configs/_base_/default_runtime.py'
]# model settings

num_classes = 7

model = dict(
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
    roi_head=dict(bbox_head=dict(num_classes=num_classes))
)
# optimizer

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')

train_dataloader = dict(
    batch_size=16,
    num_workers=4,)