#!/usr/bin/bash

python features_origin.py \
    --config_file="configs/fighting_003.yml" \
    MODEL.DEVICE_ID "('0')" \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('/data/anxiang/models/reid/fighting_003/se_resnet50_model_80.pth')" \
    OUTPUT_DIR "('./features/fighting_003_test_origin')"

python features_violet.py \
    --config_file="configs/fighting_003.yml" \
    MODEL.DEVICE_ID "('0')" \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('/data/anxiang/models/reid/fighting_003/se_resnet50_model_80.pth')" \
    OUTPUT_DIR "('./features/fighting_003_test_violet')"
