python extraect_feature_test.py \
  --config_file='configs/senet50_MGN_2.yml' \
  MODEL.DEVICE_ID "('1')" \
  DATASETS.NAMES "('dukemtmc')" \
  DATASETS.ROOT_DIR "('/home/xiangan/data_reid/testA')" \
  MODEL.PRETRAIN_CHOICE "('self')" \
  MODEL.PRETRAIN_PATH "('/mnt/anxiang/models/reid_exp/mgn_noaug/se_resnet50_model_120.pth')" \
  OUTPUT_DIR "('./exps/all_11_5_mgn')"

