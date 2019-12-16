# encoding: utf-8

import torchvision.transforms as T
import PIL
from .transforms import RandomPatch
from .transforms import RandomGaussianBlur

def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:

        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            # add by anxiang
            RandomGaussianBlur(
                prob_happen=cfg.INPUT.GAUSSIAN_BLUR_PROB,
                radius=cfg.INPUT.GAUSSIAN_BLUR_RADIUS
            ),
            # add by anxiang
            T.ColorJitter(
                brightness=cfg.INPUT.BRIGHTNESS,
                contrast=cfg.INPUT.CONTRAST,
                saturation=cfg.INPUT.SATURATION,
                hue=cfg.INPUT.HUE
            ),
            # add by anxiang
            RandomPatch(
                prob_happen=cfg.INPUT.RANDOM_PATCH_PROB,
                patch_max_area=cfg.INPUT.RANDOM_PATCH_AREA),
            T.ToTensor(),
            normalize_transform
        ])
        return transform

    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
