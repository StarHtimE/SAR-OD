_base_ = 'hrsid_deformable-detr_r50_sar.py'

model = dict(
   backbone=dict(
        # _delete_ = True,
        type='MSFA',
        use_sar=True,
        backbone=dict(
            type='ResNet',
            depth=18,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=None
        ),
        init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint='E:/Ziheng_projects/SAR-OD/checkpoints/r18.pth'),
    ), 
    neck=dict(
        type='ChannelMapper',
        in_channels=[128, 256, 512],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,)