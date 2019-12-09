#!/usr/bin/bash

# config_file
config_file=configs/${1}.yml
# output_dir
output_dir=/data/xiangan/models/reid/${1}_split4
# train_root
train_root=/root/train_split4_rematch/split4

echo "config files is ${config_file}"
echo "save path is ${output_dir}"

python tools/train.py --config_file=${config_file} \
    DATASETS.ROOT_DIR "('${train_root}')" \
    OUTPUT_DIR "('${output_dir}')" \
    MODEL.DEVICE_ID "('0,1,2,3')" \
    MODEL.PRETRAIN_PATH 'se_resnet50-ce0d4300.pth'