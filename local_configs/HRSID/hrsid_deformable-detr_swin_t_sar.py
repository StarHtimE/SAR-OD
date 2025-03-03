_base_ = 'hrsid_deformable-detr_r50_sar.py'

model = dict(
    backbone=dict(
        _delete_ = True,
        type='MSFA',
        use_sar=True, 
        backbone=dict(
            type='SwinTransformer',
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(1, 2, 3),
            with_cp=False,
            convert_weights=True,
            init_cfg=None),
        init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint='/root/workspace/SAR-OD/checkpoints/swin_t_sar_epoch_100.pth'),
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[96, 192, 384],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
)