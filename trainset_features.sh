#!/usr/bin/bash
# this shell script is used to extract
# the featuress of the training set。

# config_file
config_file=configs/${1}.yml
# output_dir
output_dir=/mnt/anxiang/models/reid_exp/${1}
# model_pretrain_path
model_pretrain_path=${output_dir}/se_resnet50_model_80.pth

echo "config files is ${config_file}"
echo "save path is ${output_dir}"

python trainset_fetures.py \
--config_file=${config_file} \
MODEL.DEVICE_ID "('0')" \
MODEL.PRETRAIN_CHOICE "('self')" \
MODEL.PRETRAIN_PATH "('${model_pretrain_path}')" \
OUTPUT_DIR "('./features/${1}')"
