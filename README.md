## 比赛结果
#### 初赛14/466名，复赛25/68名;

### 比赛比较重要的一些trick
1. Random patch，类似于Random erasing，随机再图片中添加“噪音”，不过该噪音来自训练集；
2. 比赛复赛开始我们就使用比较小的batch-size，32，我们的模型再大的batch-size上效果不好；
3. 过采样提升比较明显；
4. 观察到训练集和常规RGB图片的不同，我们再除方差的时候把GB通道反过来了；
5. Re-ranking做了一些近似，这样内存就装的下了；
6. Centerloss, Tripletloss, Softmax都用上了，center loss weight 和 triplet margin都调过参；
7. 把测试集加到验证集中当作干扰集，这样线上与线下得分一致；
8. 复赛全程用了MGN，该网络效果不错，但是速度很慢，模型也比较大，导致了我们做了很少的实验，也没法训更大的模型；
9. 将测试集通过query聚类去重，然后与gallery关联的方式加入到训练集中去训练，这个方法比较笨，但是也取得了一些效果；


## 下载预训练模型，我们仅仅用了一个预训练模型se-resnet50
http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth

## 预估训练时间，预测时间
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

## 如何运行
## 一键运行脚本run.sh即可
> # `bash run.sh`
