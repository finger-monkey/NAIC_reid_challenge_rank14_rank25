#!/usr/bin/bash

# config_file
config_file=configs/${1}.yml
# output_dir
output_dir=/data/xiangan/models/reid/${1}_train_features
# train_root
train_root=/data/xiangan/reid_final/four_fold/train_split4_rematch/all_origin
# model_path
model_path=${output_dir}/se_resnet101_model_80.pth

echo "config files is ${config_file}"
echo "save path is ${output_dir}"

python get_features_train.py \
    --config_file=${config_file} \
    MODEL.DEVICE_ID "('0,1,2,3,4,5,6,7')" \
    DATASETS.ROOT_DIR ${train_root} \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('${model_path}')" \
    OUTPUT_DIR "('./features/${1}')"


