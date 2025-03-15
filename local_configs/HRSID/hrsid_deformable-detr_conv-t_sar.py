_base_ = 'hrsid_deformable-detr_r50_sar.py'

model = dict(
    backbone=dict(
        _delete_ = True,
        type='MSFA',
        use_sar=True, 
        use_hog=False, 
        use_canny=False,
        backbone=dict(
            type='ConvNeXt',
            depths=[3, 3, 9, 3], 
            dims=[96, 192, 384, 768], 
            drop_path_rate=0.2,
            layer_scale_init_value=1e-6,
            out_indices=[0, 1, 2, 3]
        ),
        init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint='E:\Ziheng_projects\SAR-OD\checkpoints\convnext_t_sar\epoch_100.pth'),
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[96, 192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,)