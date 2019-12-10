#!/usr/bin/bash

# config_file
config_file=configs/${1}.yml
# output_dir
output_dir=/data/xiangan/models/reid/${1}_split1_8gpu
# train_root
train_root=/data/xiangan/reid_final/four_fold/train_split4_rematch/split1

echo "config files is ${config_file}"
echo "save path is ${output_dir}"

python tools/train.py --config_file=${config_file} \
    DATASETS.ROOT_DIR "('${train_root}')" \
    OUTPUT_DIR "('${output_dir}')" \
    MODEL.DEVICE_ID "('0,1,2,3,4,5,6,7')" \
    SOLVER.IMS_PER_BATCH "256"

