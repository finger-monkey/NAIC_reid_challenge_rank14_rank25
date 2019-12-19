#!/usr/bin/bash

# config_file
config_file=configs/${1}.yml
# output_dir
output_dir=/data/xiangan/models/reid/${1}_all_finetune
# train_root
train_root=/data/xiangan/reid_final/all_ex_2
# model_path
model_path=/data/xiangan/models/reid/train_005_all/se_resnet50_model_80.pth


echo "config files is ${config_file}"
echo "save path is ${output_dir}"

python tools/train.py --config_file=${config_file} \
    DATASETS.ROOT_DIR "('${train_root}')" \
    OUTPUT_DIR "('${output_dir}')" \
    MODEL.DEVICE_ID "('4,5,6,7')" \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('${model_path}')"