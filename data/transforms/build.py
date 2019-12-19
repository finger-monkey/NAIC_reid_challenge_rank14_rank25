# encoding: utf-8

import random

import torchvision.transforms as T

from .transforms import RandomPatch


class RandomRotation(object):
    def __init__(self, prob_happen, degrees):
        self.prob_happen = prob_happen
        self.degrees = degrees

    def __call__(self, img):
        if random.uniform(0, 1) > self.prob_happen:
            return img
        else:
            return T.RandomRotation(self.degrees)(img)


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN,
        std=cfg.INPUT.PIXEL_STD
    )
    if is_train:

        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            # add by anxiang
            # RandomGaussianBlur(
            #     prob_happen=cfg.INPUT.GAUSSIAN_BLUR_PROB,
            #     radius=cfg.INPUT.GAUSSIAN_BLUR_RADIUS
            # ),
            # add by anxiang
            # T.ColorJitter(
            #     brightness=cfg.INPUT.BRIGHTNESS,
            #     contrast=cfg.INPUT.CONTRAST,
            #     saturation=cfg.INPUT.SATURATION,
            #     hue=cfg.INPUT.HUE
            # ),
            # RandomRotation(
            #     prob_happen=cfg.INPUT.ROTATE_PROB,
            #     degrees=cfg.INPUT.ROTATE_DEGREE,
            # ),
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
