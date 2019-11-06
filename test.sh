#!/usr/bin/bash

config_file="configs/${1}.yml"
output_dir="('/mnt/anxiang/models/reid_exp/${1}')"

echo "config files is ${config_file}"
echo "save path is ${output_dir}"

python extract.py \
    --config_file=config_file \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.NAMES "('dukemtmc')" \
    DATASETS.ROOT_DIR "('/home/xiangan/data_reid/testA')" \
    MODEL.PRETRAIN_CHOICE "('self')" \
    MODEL.PRETRAIN_PATH "('/mnt/anxiang/models/reid_exp/${1}/se_resnet50_model_80.pth')" \
    OUTPUT_DIR "('./features/${1}')"


python get_submission.py --name "${1}"