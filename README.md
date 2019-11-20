
## 所有的配置文件都在configs中；

目前最好的结果为036，线上成绩84.6、rerank后85.7

### 在split2上训练
```shell
bash val.sh 036
```

### 在所有数据集上训练、训练完会自动是生成提交文件
```shell
bash train.sh 036
```

### 在加入query上面的split2训练
```shell
bash train_extra.sh 036
```

### 需要修改的路径

以`train.sh`举例

```xhell
#!/usr/bin/bash

# config_file
config_file=configs/${1}.yml

# output_dir 这个是要保存模型的路径，需要改动
output_dir=/mnt/anxiang/models/reid_exp/${1}

# train_root 这个是训练集的路径，需要改动
train_root=/home/xiangan/code_and_data/train_split/all

# test_root 这个是测试集的路径，需要改动
test_root=/home/xiangan/data_reid/testA

# model_pretrain_path 这个是测试的路径，不用改了
model_pretrain_path=${output_dir}/se_resnet50_model_80.pth

echo "config files is ${config_file}"
echo "save path is ${output_dir}"

python tools/train.py \
    --config_file=${config_file} \
    MODEL.DEVICE_ID "('0,1')" \  训练显卡
    DATASETS.ROOT_DIR "('${train_root}')" \
    OUTPUT_DIR "('${output_dir}')"

python extract.py \
    --config_file=${config_file} \
    MODEL.DEVICE_ID "('0')" \ 提取特征显卡
    DATASETS.ROOT_DIR ${test_root} \
    MODEL.PRETRAIN_CHOICE "('self')" \
    MODEL.PRETRAIN_PATH "('${model_pretrain_path}')" \
    OUTPUT_DIR "('./features/${1}')"

python get_submission.py --name "${1}"
```