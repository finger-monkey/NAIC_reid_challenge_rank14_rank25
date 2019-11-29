#!/usr/bin/bash

# config_file
config_file=configs/${1}.yml
# output_dir
output_dir=./${1}
# test_root
test_root=./testB
# model_pretrain_path
model_pretrain_path=${output_dir}/se_resnet101_model_80.pth

echo "config files is ${config_file}"
echo "save path is ${output_dir}"

python extract_1000.py \
    --config_file=${config_file} \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.ROOT_DIR ${test_root} \
    MODEL.PRETRAIN_CHOICE "('self')" \
    MODEL.PRETRAIN_PATH "('${model_pretrain_path}')" \
    OUTPUT_DIR "('./features_testB/${1}')"

python get_submission.py --name "${1}"
