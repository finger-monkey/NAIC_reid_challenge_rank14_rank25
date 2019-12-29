#!/usr/bin/bash

# model_path
model_path=/data/xiangan/models/reid/apex_002_split1/se_resnet50_model_80.pth

python get_features.py \
MODEL.DEVICE_ID "('0')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('${model_path}')"
