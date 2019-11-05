python tools/train.py \
  --config_file='configs/senet50_MGN_2.yml' \
  MODEL.DEVICE_ID "('0,1')" \
  DATASETS.NAMES "('dukemtmc')" \
  DATASETS.ROOT_DIR "('/home/xiangan/code_and_data/train_split/all')" \
  OUTPUT_DIR "('/mnt/anxiang/models/reid_exp/mgn_noaug')"