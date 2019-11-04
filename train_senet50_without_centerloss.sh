python tools/train.py \
  --config_file='configs/senet50_without_centerloss.yml' \
  MODEL.DEVICE_ID "('4')" \
  DATASETS.NAMES "('dukemtmc')" \
  DATASETS.ROOT_DIR "('/home/xiangan/code_and_data/train_split/all')" \
  OUTPUT_DIR "('/mnt/anxiang/models/reid_exp/all_11_4_withoutcenterloss')"