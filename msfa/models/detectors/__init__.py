# Copyright (c) OpenMMLab. All rights reserved.
from .distill_deformable_detr import DistillDeformableDETR
from .deformable_detr_querycross import QueryCrossDeformableDETR

__all__ = [
    'DistillDeformableDETR', 'QueryCrossDeformableDETR'
]
