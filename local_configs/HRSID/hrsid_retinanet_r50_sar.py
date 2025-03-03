_base_ = [
    '/root/workspace/SAR-OD/configs/_base_/models/retinanet_r50_fpn.py', 
    '/root/workspace/SAR-OD/configs/_base_/datasets/HRSID.py',
    '/root/workspace/SAR-OD/configs/_base_/schedules/schedule_1x.py', '/root/workspace/SAR-OD/configs/_base_/default_runtime.py'
]# model settings

num_classes = 1

model = dict(
    type='RetinaNet',
    backbone=dict(
        _delete_ = True,
        type='MSFA',
        use_sar=True,
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=None
        ),
        init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint='/root/workspace/SAR-OD/checkpoints/r50_sar_epoch_100.pth'),
    ),
    
    bbox_head=dict(
        num_classes=num_classes,)
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