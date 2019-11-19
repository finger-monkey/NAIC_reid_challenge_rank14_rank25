# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing
from .transforms import Random2DTranslation


def build_transforms(cfg, is_train=True):
    #
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    #
    if cfg.INPUT.CROP == 'norm':
        CROP = T.RandomCrop(cfg.INPUT.SIZE_TRAIN)
    elif cfg.INPUT.CROP == 'random':
        CROP = Random2DTranslation(
            height=cfg.INPUT.SIZE_TRAIN[0],
            width=cfg.INPUT.SIZE_TRAIN[1])
    else:
        raise ValueError

    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            CROP(cfg.INPUT.SIZE_TRAIN),
            # RandomSizedRectCrop(height=384, width=128),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=(0.0972, 0.1831, 0.2127))
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
