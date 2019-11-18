# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
from .mgn2048 import MGN_2048


def build_model(cfg, num_classes):

    if cfg.MODEL.MGN_2048 == 'yes':
        print("using model MGN2048")
        model = MGN_2048(
            num_classes=num_classes,
            last_stride=cfg.MODEL.LAST_STRIDE,
            model_path=cfg.MODEL.PRETRAIN_PATH,
            model_name=cfg.MODEL.NAME,
            pretrain_choice=cfg.MODEL.PRETRAIN_CHOICE,
        )
        return model
    else:
        print("using baseline")
        model = Baseline(
            num_classes=num_classes,
            last_stride=cfg.MODEL.LAST_STRIDE,
            model_path=cfg.MODEL.PRETRAIN_PATH,
            model_name=cfg.MODEL.NAME,
            pretrain_choice=cfg.MODEL.PRETRAIN_CHOICE,
            reduction=cfg.MODEL.REDUCTION
        )
        return model
