#!/usr/bin/bash

# origin
python features_origin.py \
    --config_file=configs/model_base.yml \
    MODEL.DEVICE_ID "('0')" \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('/tmp/data/model/model_base/se_resnet50_model_60.pth')" \
    OUTPUT_DIR "('/tmp/data/features/origin')"

