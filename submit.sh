#!/usr/bin/bash

# config_file
config_file=configs/${1}.yml
# output_dir
output_dir=/data/xiangan/models/reid/${1}
# train_root
train_root=/data/xiangan/reid_data/trainset
# model_path
model_path=${output_dir}/se_resnet50_model_40.pth

echo "config files is ${config_file}"
echo "save path is ${output_dir}"


python get_features.py \
    --config_file=${config_file} \
    MODEL.DEVICE_ID "('7,8')" \
    DATASETS.ROOT_DIR ${train_root} \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('${model_path}')" \
    OUTPUT_DIR "('./features_test/${1}')"

python get_submission.py --name "${1}"
