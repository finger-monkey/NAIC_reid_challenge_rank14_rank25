#!/usr/bin/bash

# config_file
config_file=configs/model_final.yml
# train_root
train_root=/tmp/data/all
# model_path
model_path=/tmp/data/model/model_final/se_resnet50_model_75.pth


python get_features.py \
    --config_file=${config_file} \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.ROOT_DIR ${train_root} \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('${model_path}')" \
    OUTPUT_DIR "('/tmp/data/features/model_final')"


