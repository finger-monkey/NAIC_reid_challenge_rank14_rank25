# Experiment
### random erasing(online)  

| erasing_prob |        1       |        2        |  3  |
|:------------:|:--------------:|:---------------:|:---:|
|     0.05     |      80.74     |        -        |  -  |
|     0.10     |      81.66     |        -        |  -  |
|     0.15     |      81.69     |        -        |  -  |
|     0.20     | 81.47(anxiang) | 82.99(yongfeng) |  -  |
|     0.25     |      80.83     |        -        |  -  |
|     0.30     |      82.10     |        -        |  -  |
|     0.50     |        -       |        -        |  -  |
|   baseline   |      81.40     |        -        |  -  |


### random erasing repeat prob 0.2(online)

| erasing_prob |   1   |   2   |   3   |   4   |
|:------------:|:-----:|:-----:|:-----:|:-----:|
|     0.20     | 80.74 | 82.26 | 81.78 | 81.11 |


### 2019 11.12 baseline(dev-set)
train-dev set is split2

| experiment | rank-1 |  mAP | online |
|:----------:|:------:|:----:|:------:|
|      1     |  90.2  | 80.8 |  81.4  |
|      2     |  90.4  | 80.7 |    -   |
|      3     |  89.5  | 80.2 |    -   |
|   average  |  90.0  | 80.6 |    -   |

### 2019 11.13 
### `raranking k1=7,k2=3,lambda=0.85`
|     experiment     | rank-1(dev) | mAP(dev) | online | reranking |
|:------------------:|:-----------:|:--------:|:------:|:---------:|
|       +sample      |     90.7    |   82.2   |  82.92 |   84.12   |
|      baseline      |     90.4    |   80.7   |  81.40 |   82.60   |
|   +sample +bnneck  |     86.9    |   76.1   |    -   |     -     |
| +sample +focalloss |     90.7    |   82.3   |  82.42 |   83.81   |

### 2019 11.14
### 分层学习率，FC部分 乘 10
|  experiment  | rank-1 |  mAP | online | reranking |
|:------------:|:------:|:----:|:------:|:---------:|
|    lr\*10    |  91.6  | 83.6 |  83.62 |   84.67   |
|   baseline   |  90.7  | 82.2 |  82.92 |   84.12   |
| lr\*10 fixbn |  91.3  | 83.0 |  82.60 |     -     |

### 2019 11.15
### sample 3x 
### random erasing 0.2
### random sized rect crop

### 2019 11.16
|             experiment            | rank-1 |  mAP | online | reranking |
|:---------------------------------:|:------:|:----:|:------:|:---------:|
|              baseline             |  90.7  | 82.2 |  82.92 |   84.12   |
|               分层学习率               |  91.6  | 83.6 |  83.62 |   84.67   |
|      random erasing 0.2 实验 1      |  91.6  | 84.7 |  84.19 |   85.23   |
| random erasing 0.2 + query 对比实验 1 |  91.6  | 84.0 |  84.60 |   85.43   |
|   random erasing 0.2 实验 2 对比实验 1  |    -   |   -  |  84.51 |   86.05   |

# 在训练集合中加测试集query的方式：
先算了一下query之间的距离，距离得分超过0.6的都去掉、大概去晒去10%的query;

# 需要做的实验
- [ ] 让random erasing 稳定提点；
- [ ] 在global feature 中采用随机抹去的实验；
- [ ] 测试增强；
- [x] 目前只加入了测试集中的query、还没有加入gallery；
- [x] random crop的方式需要再考虑下，分析得出黑色应该是有语义的，直接pad0应该不对、可以pad均值，或者采用resize1.25再random crop的方式；
- [ ] Efficient的加入，更好的baseline模型；   
https://github.com/lukemelas/EfficientNet-PyTorch
- [ ] k倒排编码也发现有着随机提点的情况，最高可以提高1.4个点，最低是0.5个点；
- [ ] 颜色相关的数据增强还没有实现；
- [x] RandomPatch数据增强的实现，维护一个patch pool，然后再数据增强的时候随机选择一个patch粘贴再原图上，来模拟遮挡，代码有现成的;  
https://github.com/KaiyangZhou/deep-person-reid/blob/099b0ae7fcead522e56228860221a4f8b06cdaad/torchreid/data/transforms.py#L134
- [ ] 模型融合；
- [ ] 训练集的清洗，消除掉类很相近的id、删除掉离群的样本；

### 2019 11.17

# 加入query的实验

|   experiment  | rank-1 |  mAP | online | reranking |
|:-------------:|:------:|:----:|:------:|:---------:|
|  +query(034)  |  91.5  | 82.8 |  83.73 |     -     |
| baseline(021) |  91.6  | 83.6 |  83.62 |   84.67   |

# random crop的实验

|    experiment    | rank-1 |  mAP | online | reranking |
|:----------------:|:------:|:----:|:------:|:---------:|
|   baseline(021)  |  91.6  | 83.6 |  83.62 |   84.67   |
| random crop(035) |  90.5  | 82.8 |  82.31 |     -     |

# random patch 的实验

|          experiment          | rank-1 |  mAP | online | reranking |
|:----------------------------:|:------:|:----:|:------:|:---------:|
|         baseline(021)        |  91.6  | 83.6 |  83.62 |   84.67   |
|       random patch(036)      |  91.8  | 83.8 |  84.60 |   85.73   |
|  random patch(040) prob 0.3  |    -   |   -  |  84.83 |     -     |
|  random patch(041) prob 0.4  |    -   |   -  |  84.95 |     -     |
| random patch(038) repeat 036 |    -   |   -  |  84.48 |     -     |
| random patch(039) repeat 036 |    -   |   -  |  84.92 |     -     |


<!-- # batch-dropblock 的实验 (没啥用)

|   experiment  | rank-1 | mAP | online | reranking |     |
|:-------------:|:------:|:---:|:------:|:---------:|-----|
|      042      |    -   |  -  |    -   |     -     |     |
|      043      |    -   |  -  |    -   |     -     |     |
|      045      |    -   |  -  |    -   |     -     |     |
|      046      |    -   |  -  |    -   |     -     |     |
|      047      |    -   |  -  |    -   |     -     |     |
|      048      |    -   |  -  |    -   |     -     |     |
|      049      |    -   |  -  |    -   |     -     | --> |
| baseline(040) |    -   |  -  |  84.83 |   86.00   |     |


# gallery clean (添加清洗后的测试集、query去重阈值0.6、gallary关联阈值0.7)
|    experiment   | rank-1 |  mAP | online | reranking |
|:---------------:|:------:|:----:|:------:|:---------:|
|       050       |  92.5  | 85.0 |    -   |     -     |
|       051       |  92.4  | 85.4 |    -   |     -     |
|       052       |  91.3  | 84.1 |    -   |     -     |
|       053       |  91.8  | 85.4 |    -   |     -     |
|       054       |    -   |   -  |  85.54 |     -     |
|       055       |    -   |   -  |  86.18 |     -     |
|       056       |    -   |   -  |  86.31 |   87.00   |
|  baseline(036)  |  91.8  | 83.8 |  84.83 |   86.00   |
| 054 + 055 + 056 |    -   |   -  |    -   |   87.50   |


# random rotate 
|   experiment  | rank-1 | mAP | online | reranking |
|:-------------:|:------:|:---:|:------:|:---------:|
| baseline(056) |    -   |  -  |  86.31 |   87.00   |
|      102      |    -   |  -  |  85.91 |   86.99   |
|      102      |    -   |  -  |    -   |     -     |
|      103      |    -   |  -  |    -   |     -     |
|      104      |    -   |  -  |    -   |     -     |
|      105      |    -   |  -  |    -   |     -     |


# remove repeat
|     experiment     | rank-1 | mAP | online | reranking |
|:------------------:|:------:|:---:|:------:|:---------:|
|    baseline(056)   |    -   |  -  |  86.31 |   87.00   |
| theshold=0.96(300) |    -   |  -  |  85.34 |     -     |
| theshold=0.96(301) |    -   |  -  |  86.05 |     -     |
| theshold=0.97(302) |    -   |  -  |  86.30 |     -     |
| theshold=0.97(303) |    -   |  -  |  86.10 |     -     |


# increase epoch
```yml
SOLVER:
    MAX_EPOCHS: 160
    STEPS: [80, 120, 140]
```
|   experiment  | rank-1 | mAP | online | reranking |
|:-------------:|:------:|:---:|:------:|:---------:|
| baseline(056) |    -   |  -  |  86.31 |   87.00   |
|     (113)     |    -   |  -  |  86.39 |     -     |
|     (114)     |    -   |  -  |  86.34 |     -     |
|     (115)     |    -   |  -  |  86.15 |     -     |


# triplet margin 1.2
```yml
SOLVER:
    MARGIN: 1.20
```
|   experiment  | rank-1 | mAP | online | reranking |
|:-------------:|:------:|:---:|:------:|:---------:|
| baseline(056) |    -   |  -  |  86.31 |   87.00   |
|     (402)     |    -   |  -  |  86.40 |     -     |
|     (404)     |    -   |  -  |  85.60 |     -     |
|     (405)     |    -   |  -  |  86.74 |     -     |