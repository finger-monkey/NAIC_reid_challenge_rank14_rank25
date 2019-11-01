python tools/extract_feature_query.py \
--config_file='configs/softmax_triplet_with_center_anxiang.yml' \
MODEL.DEVICE_ID "('1')" \
DATASETS.NAMES "('dukemtmc')" \
DATASETS.ROOT_DIR "('/home/xiangan/code_and_data/train_split/split2')" \
MODEL.PRETRAIN_CHOICE "('self')" \
MODEL.PRETRAIN_PATH "('/mnt/anxiang/models/reid_exp/experiment2/se_resnet101_center_param_120.pth')" \
OUTPUT_DIR "('./exps/test')"
