#!/usr/bin/bash

# config_file
config_file=configs/${1}.yml
# output_dir
output_dir=/gpu/data2/deepglint/models${1}_all
# train_root
train_root=/gpu/data2/deepglint/dgreid/dataset/train_split4_rematch/all

echo "config files is ${config_file}"
echo "save path is ${output_dir}"

python tools/train.py --config_file=${config_file} \
    DATASETS.ROOT_DIR "('${train_root}')" \
    OUTPUT_DIR "('${output_dir}')" \
    MODEL.DEVICE_ID "('0,1,2,3')"