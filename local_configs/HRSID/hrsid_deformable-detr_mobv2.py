_base_ = 'hrsid_deformable-detr_r50_sar.py'

model = dict(
    backbone=dict(
        _delete_ = True,
        type='MobileNetV2',
        out_indices=(2, 4, 7),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        act_cfg=dict(type='ReLU6'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://mobilenet_v2')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[32, 96, 1280],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
)

train_dataloader = dict(
    batch_size=16,
    num_workers=4,)