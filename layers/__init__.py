# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .arcface_loss import ArcfaceLoss
from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .center_loss import CenterLoss
from .focal_loss import FocalLoss


def make_loss(cfg, num_classes):  # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    else:
        raise ValueError

    if cfg.MODEL.IF_WITH_ARC_FACE_LOSS == 'yes':
        arcface = ArcfaceLoss(
            embedding_size=2048,
            class_num=num_classes,
            s=64,
            m=cfg.SOLVER.ARC_FACE_MARGIN
        )

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        raise ValueError
        # xent = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo
        # print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        raise ValueError
        # def loss_func(score, feat, target):
        #     return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        raise ValueError
        # def loss_func(score, feat, target):
        #     return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            # remove final feature
            final_feature = feat[-1]
            feat = feat[0:3]
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    raise ValueError
                else:
                    if cfg.MODEL.IF_WITH_ARC_FACE_LOSS == 'yes':

                        # get arc-face feat
                        arcface_feat = arcface(final_feature, target)
                        return sum(F.cross_entropy(s, target) for s in score) + \
                               sum(triplet(f, target)[0] for f in feat) + \
                               F.cross_entropy(arcface_feat, target)

                    elif cfg.MODEL.IF_WITH_ARC_FACE_LOSS == 'no':
                        return sum(F.cross_entropy(s, target) for s in score) + \
                               sum(triplet(f, target)[0] for f in feat)
                    else:
                        raise ValueError
            else:
                raise ValueError
    else:
        raise ValueError
        # print('expected sampler should be softmax, triplet or softmax_triplet, '
        #       'but got {}'.format(cfg.DATALOADER.SAMPLER))

    return loss_func


def make_loss_with_center(cfg, num_classes):  # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048

    if cfg.SOLVER.FOCAL_LOSS == 'yes':
        flce = FocalLoss(
            class_num=num_classes,
            alpha=None,
            gamma=cfg.SOLVER.FOCAL_LOSS_GAMMA,
            size_average=True)

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        final_feature = feat[-1]
        feat = feat[0:3]
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                raise ValueError
            else:
                raise ValueError

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                raise ValueError
            elif cfg.SOLVER.FOCAL_LOSS == 'yes':
                return sum(flce(s, target) for s in score) + \
                       sum(triplet(f, target)[0] for f in feat) + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(final_feature, target)
            else:
                return sum(F.cross_entropy(s, target) for s in score) + \
                       cfg.SOLVER.TRIPLET_LOSS_WEIGHT * sum(triplet(f, target)[0] for f in feat) + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(final_feature, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    return loss_func, center_criterion
