#!/usr/bin/bash

# config_file
config_file=configs/dgreid_test_003.yml

# model_path
model_path=/data/anxiang/models/reid/dgreid_test_003_split1/se_resnet50_model_80.pth

echo "config files is ${config_file}"

python get_features_val.py \
    --config_file=${config_file} \
    MODEL.DEVICE_ID "('0')" \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('${model_path}')" \
    OUTPUT_DIR "('./features/${1}')"
