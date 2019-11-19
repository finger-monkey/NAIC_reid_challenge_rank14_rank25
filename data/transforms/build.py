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

    if is_train:
        if cfg.INPUT.CROP == 'norm':
            print('using RandomCrop...')
            transform = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                normalize_transform,
                RandomErasing(probability=cfg.INPUT.RE_PROB, mean=(0.0972, 0.1831, 0.2127))
            ])
        elif cfg.INPUT.CROP == 'random':
            print('using Random2DTranslation')
            transform = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                Random2DTranslation(height=cfg.INPUT.SIZE_TRAIN[0], width=cfg.INPUT.SIZE_TRAIN[1]),
                T.ToTensor(),
                normalize_transform,
                RandomErasing(probability=cfg.INPUT.RE_PROB, mean=(0.0972, 0.1831, 0.2127))
            ])
        else:
            raise ValueError
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
