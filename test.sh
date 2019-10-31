python tools/test.py \
  --config_file='configs/softmax_triplet_with_center_anxiang.yml' \
  MODEL.DEVICE_ID "('2')" \
  DATASETS.NAMES "('dukemtmc')" \
  DATASETS.ROOT_DIR "('/home/xiangan/code_and_data/train_split/split2')" \
  MODEL.IF_WITH_CENTER "('yes')" \
  MODEL.METRIC_LOSS_TYPE "('triplet_center')" \
  TEST.NECK_FEAT "('after')" \
  TEST.FEAT_NORM "('yes')" \
  TEST.WEIGHT "('/mnt/anxiang/models/reid_exp/experiment2/se_resnet101_center_param_120.pth')" \
  OUTPUT_DIR "('./exps/test')"
