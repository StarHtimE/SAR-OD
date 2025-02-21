_base_ = [
    '../../../configs/_base_/models/faster-rcnn_r50_fpn.py', 
    '../../../configs/_base_/datasets/SARDet_100k.py', # sar_detection   sar_objectness
    '../../../configs/_base_/schedules/schedule_1x.py', '../../../configs/_base_/default_runtime.py'
]# model settings
num_class = 6
# model settings
model = dict(
    backbone=dict(
        _delete_ = True,
        type='MSFA',
        use_sar=True, 
        backbone=dict(
            type='SwinTransformer',
            embed_dims=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            convert_weights=True,
            init_cfg=None),
        init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint='/root/siton-gpfs-archive/yuxuanli/mmpretrain/work_dirs/swin_s_sar/epoch_100.pth'),

    ), 
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        bbox_head=dict(
            num_classes=num_class,)),
)


optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
