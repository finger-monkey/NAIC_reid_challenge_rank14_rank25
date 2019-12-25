#!/usr/bin/bash


config_file="configs/jiankang_train_001.yml"
dataset="/data/xiangan/reid_final/all"
model=/data/xiangan/models/reid/jiankang_train_001_all/se_resnet101_model_80.pth

python get_features_origin.py \
    --config_file=$config_file \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.ROOT_DIR  $dataset \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('${model}')" \
    OUTPUT_DIR "('./features/jiankang_train_001_test_origin')"


python get_features_violet.py \
    --config_file=$config_file \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.ROOT_DIR $dataset \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('${model}')" \
    OUTPUT_DIR "('./features/jiankang_train_001_test_violet')"


