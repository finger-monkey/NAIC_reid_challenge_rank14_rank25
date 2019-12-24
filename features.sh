#!/usr/bin/bash

python get_features_origin.py \
    --config_file=configs/apex_003.yml \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.ROOT_DIR /data/xiangan/reid_final/all \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('/data/xiangan/models/reid/apex_003/se_resnet50_model_80.pth')" \
    OUTPUT_DIR "('./features/apex_003_test_origin')"


python get_features_violet.py \
    --config_file=configs/violet_003.yml \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.ROOT_DIR /data/xiangan/reid_final/violet_another \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('/data/xiangan/models/reid/violet_003_all/se_resnet50_model_80.pth')" \
    OUTPUT_DIR "('./features/apex_003_test_violet')"


