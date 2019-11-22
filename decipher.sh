#
output_dir=/mnt/anxiang/models/reid_exp/054
model_pretrain_path=${output_dir}/se_resnet50_model_80.pth

python statistics/decipher.py \
    --config_file configs/054.yml \
    MODEL.DEVICE_ID "('0')" \
    MODEL.PRETRAIN_CHOICE "('self')" \
    MODEL.PRETRAIN_PATH "('${model_pretrain_path}')"
