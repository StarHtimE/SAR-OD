# 基础配置继承（保留原有结构）
_base_ = [
    '/workspace/SARDet_100K/MSFA/configs/_base_/datasets/HRSID.py',  # 指向你的 HRSID 数据集配置文件
    '/workspace/SARDet_100K/MSFA/configs/_base_/schedules/schedule_1x.py',
    '/workspace/SARDet_100K/MSFA/configs/_base_/default_runtime.py'
]

# 模型配置
model = dict(
    type='RetinaNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],  # 保持 COCO 均值，如需更改需重新计算
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,  # 可改为 0 解冻所有层进行微调
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,  # 关键修改！HRSID 只有船舶一类
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=3,  # 原为4，减小基础尺寸适应小目标
            scales_per_octave=5,  # 原为3，增加尺度密度
            ratios=[0.25, 0.5, 1.0, 2.0, 4.0],  # 扩展宽高比范围
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),  # 调整编码标准差
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=3.0,  # 原为2.0，加强对困难样本的关注
            alpha=0.6,  # 原为0.25，平衡正负样本
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss',  # 原为L1，改用平滑L1
            beta=0.11,  # 平滑参数
            loss_weight=1.2)),  # 增加定位损失权重
    # 训练/测试参数优化
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.4,  # 原为0.5，降低正样本阈值
            neg_iou_thr=0.3,  # 原为0.4，减少模糊样本
            min_pos_iou=0.1,  # 原为0，允许低质量匹配
            ignore_iof_thr=0.5),  # 忽略与忽略区域重叠的锚框
        sampler=dict(
            type='RandomSampler',  # 原为PseudoSampler，改用随机采样
            num=512,
            pos_fraction=0.4,  # 正样本比例
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,  # 原为-1，禁止边界锚框
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,  # 原为1000，增加预选框数量
        min_bbox_size=5,  # 原为0，过滤极小预测框
        score_thr=0.01,  # 原为0.05，降低阈值保留更多候选
        nms=dict(type='soft_nms', iou_threshold=0.4, method='gaussian'),  # 改用软NMS
        max_per_img=300))  # 原为100，增加最大检测数





