_base_ = [
    '/workspace/SAR-OD/configs/_base_/models/faster-rcnn_r50_fpn.py', 
    '/workspace/SAR-OD/configs/_base_/datasets/HRSID.py',
    '/workspace/SAR-OD/configs/_base_/schedules/schedule_1x.py', '/workspace/SAR-OD/configs/_base_/default_runtime.py'
]# model settings

num_class = 1
model = dict( 
    backbone=dict(
        _delete_ = True,
        type='MSFA',
        use_sar=True, 
        # input_size = (512,512), 
        backbone=dict(
            type='VAN',
            embed_dims=[64, 128, 320, 512],
            drop_rate=0.1,
            drop_path_rate=0.1,
            depths=[3, 3, 12, 3],
            norm_cfg=dict(type='SyncBN', requires_grad=True)),
        init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint='/workspace/SAR-OD/checkpoints/van_b_sar_wavelet_epoch_100.pth'),
    ), 
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        bbox_head=dict(
            num_classes=num_class,)),
)


param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]


optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        betas=(
            0.9,
            0.999,
        ), lr=0.0002, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')

train_dataloader = dict(
    batch_size=4,
    num_workers=4,)
