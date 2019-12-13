#!/usr/bin/bash

# config_file
config_file=configs/${1}.yml
# output_dir
output_dir=/train/trainset/1/models/reid_exp/${1}
# train_root
train_root=/root/train_split4_rematch

for split in {"split1","split2","split3","split4"}; do

    echo "config files is ${config_file}"
    echo "save path is ${output_dir}_${split}"
    echo "train path is ${train_root}/${split}"

    python tools/train.py --config_file=${config_file} \
    DATASETS.ROOT_DIR "('${train_root}/${split}')" \
    OUTPUT_DIR "('${output_dir}_${split}')" \
    MODEL.DEVICE_ID "('0,1,2,3')" \
    SOLVER.IMS_PER_BATCH "256" \
    MODEL.PRETRAIN_PATH 'se_resnet50-ce0d4300.pth'
done


