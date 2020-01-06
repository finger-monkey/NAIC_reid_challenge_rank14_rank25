# 比赛复现指北

# 下载预训练模型，我们仅仅用了一个预训练模型se-resnet50
http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth

# 预估训练时间，预测时间
### 1. 训练时间：44小时；
### 2. 预测：2小时；  

| 步骤         | 时间/h    |
|------------|---------|
| 训练中间模型     | 12 小时   |
| 推理测试集特征    | 0.23 小时 |
| 无监督聚类      | 0.14 小时 |
| 训练最终模型     | 32 小时   |
| 最终模型提取特征时间 | 0.2 小时  |
| reranking  | 2小时     |
| total      | 47      |

## DockerFile

详情见`./DockerFile`.
1. 我们训练和推理的容器为同一个容器；
2. 项目在容器中的路径为：`/workspace/code`
3. 我们checkpoint都存放在了`/tmp/data/model`路径下；
4. 最终的模型为`/tmp/data/model/model_final/se_resnet50_model_75.pth`;

# 如何运行
## 一键运行脚本run.sh即可
> # `bash run.sh`
