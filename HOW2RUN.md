### 
1. 线下 `bash val.sh baseline` baseline config文件名
2. 线上 `bash train.sh baseline`
3. 提取测试集特征、生成提交文件 `bash submit.sh baseline`

### val.sh
```shell script
#!/usr/bin/bash
# config_file
config_file=configs/${1}.yml # config 文件
# output_dir
output_dir=/data/xiangan/models/reid/${1}_val # 模型保存的地址
# train_root
train_root=/data/xiangan/reid_data/split2 # 训练集地址

echo "config files is ${config_file}"
echo "save path is ${output_dir}"

python tools/train.py --config_file=${config_file} \
    DATASETS.ROOT_DIR "('${train_root}')" \
    OUTPUT_DIR "('${output_dir}')" \
    MODEL.DEVICE_ID "('1,2')" # 用哪几块卡train
```


### submit.sh
```shell script
#!/usr/bin/bash

# config_file
config_file=configs/${1}.yml # config 文件
# output_dir
output_dir=/data/xiangan/models/reid/${1}
# train_root
train_root=/data/xiangan/reid_data/trainset # 训练集地址
# model_path
model_path=${output_dir}/se_resnet50_model_40.pth # 模型地址

echo "config files is ${config_file}"
echo "save path is ${output_dir}"

# 测试集地址写死在get_features.py
python get_features.py \
    --config_file=${config_file} \
    MODEL.DEVICE_ID "('7,8')" \ # 提取特征的卡
    DATASETS.ROOT_DIR ${train_root} \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.WEIGHT "('${model_path}')" \
    OUTPUT_DIR "('./features_test/${1}')" # 存放特征的地址

python get_submission.py --name "${1}" 
```

