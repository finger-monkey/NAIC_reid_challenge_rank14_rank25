#!/usr/bin/bash


config_file="configs/apex_003.yml"
dataset="/data/xiangan/reid_final/all"
model=/data/xiangan/models/reid/apex_003_all/se_resnet50_model_80.pth

python get_features_origin.py \
    --config_file= $config_file \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.ROOT_DIR  $dataset \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('${model}')" \
    OUTPUT_DIR "('./features/apex_003_test_origin')"


python get_features_violet.py \
    --config_file=$config_file \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.ROOT_DIR $dataset \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('${model}')" \
    OUTPUT_DIR "('./features/apex_003_test_violet')"


