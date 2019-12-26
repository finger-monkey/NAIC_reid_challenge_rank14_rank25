#!/usr/bin/bash

# config_file
config_file=configs/${1}.yml
# output_dir
output_dir=/raid/jiankangdeng/reid/models/${1}_all_12_27
# train_root
train_root=/raid/jiankangdeng/reid/train_split4_rematch/all_12_27

echo "config files is ${config_file}"
echo "save path is ${output_dir}"

python tools/train.py --config_file=${config_file} \
    DATASETS.ROOT_DIR "('${train_root}')" \
    OUTPUT_DIR "('${output_dir}')" \
    MODEL.DEVICE_ID "('0')"