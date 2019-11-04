python tools/extract_feature_all.py \
--config_file='configs/softmax_triplet_with_center_anxiang.yml' \
MODEL.DEVICE_ID "('1')" \
DATASETS.NAMES "('dukemtmc')" \
DATASETS.ROOT_DIR "('/home/xiangan/code_and_data/train_split/all')" \
MODEL.PRETRAIN_CHOICE "('self')" \
MODEL.PRETRAIN_PATH "('/mnt/anxiang/models/reid_exp/all/se_resnet101_model_120.pth')" \
OUTPUT_DIR "('./exps/features')"
