# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
from .bnneck_mgn import BNNeck_MGN


def build_model(cfg, num_classes):
    if cfg.MODEL.BNNECK == 'yes':
        model = BNNeck_MGN(
            num_classes=num_classes,
            last_stride=cfg.MODEL.LAST_STRIDE,
            model_path=cfg.MODEL.PRETRAIN_PATH,
            model_name=cfg.MODEL.NAME,
            pretrain_choice=cfg.MODEL.PRETRAIN_CHOICE,
            reduction=cfg.MODEL.REDUCTION
        )
        return model
    elif cfg.MODEL.BNNECK == 'no':
        model = Baseline(
            num_classes=num_classes,
            last_stride=cfg.MODEL.LAST_STRIDE,
            model_path=cfg.MODEL.PRETRAIN_PATH,
            model_name=cfg.MODEL.NAME,
            pretrain_choice=cfg.MODEL.PRETRAIN_CHOICE,
            reduction=cfg.MODEL.REDUCTION
        )
        return model
    else:
        raise ValueError
