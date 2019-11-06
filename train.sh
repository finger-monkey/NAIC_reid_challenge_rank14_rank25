config_file="configs/${1}.yml"
output_dir="('/mnt/anxiang/models/reid_exp/${1}')"

echo "config files is ${config_file}"
echo "save path is ${output_dir}"

python tools/train.py \
    --config_file=config_file \
    MODEL.DEVICE_ID "('0,1')" \
    DATASETS.NAMES "('dukemtmc')" \
    DATASETS.ROOT_DIR "('/home/xiangan/code_and_data/train_split/all')" \
    OUTPUT_DIR output_dir
