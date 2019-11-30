#!/usr/bin/bash

# config_file
config_file=configs/${1}.yml
# output_dir
output_dir=/mnt/anxiang/models/reid_exp/${1}_val
# train_root
train_root=/data/xiangan/reid_data/split2

echo "config files is ${config_file}"
echo "save path is ${output_dir}"

python tools/train.py --config_file=${config_file} \
    DATASETS.ROOT_DIR "('${train_root}')" \
    OUTPUT_DIR "('${output_dir}')"