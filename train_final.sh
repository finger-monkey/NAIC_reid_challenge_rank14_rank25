#!/usr/bin/bash

config_file=configs/model_final.yml
train_root=/tmp/data/all

echo "config files is ${config_file}"

python tools/train.py --config_file=${config_file} DATASETS.ROOT_DIR "('${train_root}')" MODEL.DEVICE_ID "('0')"