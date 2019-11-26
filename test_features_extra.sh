#!/usr/bin/bash

# config_file
config_file=configs/${1}.yml
# output_dir
output_dir=/mnt/anxiang/models/reid_exp/${1}
# test_root
test_root=/home/xiangan/data_reid/testA
# model_pretrain_path
model_pretrain_path=${output_dir}/se_resnet50_model_80.pth

echo "config files is ${config_file}"
echo "save path is ${output_dir}"


python test_features_extra.py \
    --config_file=${config_file} \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.ROOT_DIR ${test_root} \
    MODEL.PRETRAIN_CHOICE "('self')" \
    MODEL.PRETRAIN_PATH "('${model_pretrain_path}')" \
    OUTPUT_DIR "('./features_flip/${1}')"

python get_submission_flip.py --name "${1}"