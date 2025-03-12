_base_ = 'hrsid_deformable-detr_r50_sar.py'

model = dict(
    backbone=dict(
        _delete_ = True,
        type='MSFA',
        use_sar=True, 
        # input_size = (512,512), 
        backbone=dict(
            type='VAN',
            embed_dims=[128, 320, 512],
            drop_rate=0.1,
            drop_path_rate=0.1,
            depths=[3, 3, 12, 3],
            norm_cfg=dict(type='SyncBN', requires_grad=True)),
        init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint='E:/Ziheng_projects/SAR-OD/checkpoints/van_b_sar_wavelet_epoch_100.pth'),
    ), 
    neck=dict(
        type='ChannelMapper',
        in_channels=[128, 320, 512],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,)