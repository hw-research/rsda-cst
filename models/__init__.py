"""
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import os
from functools import partial

from .deeplabv2 import DeepLabV2_ResNet101, DeepLabV2_VGG16
from .fcn import VGG16_FCN8s
from .deeplabv3plus import DeepLabV3Plus
from .sac import SAC, SAC_Baseline


def get_model(cfg, rank, *args, **kwargs):
    """
    Create model based on configuration.

    Args:
        cfg: global configuration object (from core.config)
        rank: current process rank (for distributed training)
        *args, **kwargs: additional arguments (e.g., num_classes, criterion)

    Returns:
        SAC_Baseline or SAC model wrapped with DistributedDataParallel.
    """
    # Model registry
    models = {
        'deeplabv2_resnet101': DeepLabV2_ResNet101,
        'deeplabv2_vgg16_bn': partial(DeepLabV2_VGG16, use_bn=True),
        'fcn_vgg16_bn': partial(VGG16_FCN8s, use_bn=True),
        'deeplabv3plus_resnet50': partial(DeepLabV3Plus, backbone='resnet50'),
        'deeplabv3plus_resnet101': partial(DeepLabV3Plus, backbone='resnet101'),
    }

    arch = cfg.MODEL.ARCH.lower()
    # Prepare backbone keyword arguments
    backbone_kwargs = {
        'num_classes': kwargs.get('num_classes'),
        'criterion': kwargs.get('criterion'),
    }
    if arch.startswith('deeplabv3plus_'):
        backbone_kwargs['aspp_rates'] = cfg.MODEL.ASPP_RATES
        backbone_kwargs['output_stride'] = cfg.MODEL.OUTPUT_STRIDE
    else:
        backbone_kwargs['freeze_bn'] = not cfg.MODEL.BASELINE

    # Manual pretrained weight loading
    if hasattr(cfg.MODEL, 'INIT_MODEL') and cfg.MODEL.INIT_MODEL and os.path.isfile(cfg.MODEL.INIT_MODEL):
        backbone_kwargs['pretrained'] = cfg.MODEL.INIT_MODEL
        print(f"Using pretrained weights from: {cfg.MODEL.INIT_MODEL}")
    else:
        print(f"Pretrained weights not found: {cfg.MODEL.INIT_MODEL}. Backbone will be randomly initialized.")
        backbone_kwargs['pretrained'] = None

    # Create backbone
    backbone = models[arch](**backbone_kwargs)

    if cfg.MODEL.BASELINE:
        # Baseline mode: only source domain training
        return SAC_Baseline(cfg, backbone, rank, **kwargs)
    else:
        # Self-supervised mode: create teacher network (slow copy) and wrap with SAC
        slow_copy = models[arch](**backbone_kwargs)
        return SAC(cfg, backbone, slow_copy, rank, **kwargs)
