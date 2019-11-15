# Experiment
### random erasing(online)  

| erasing_prob |       1        |        2        | 3 |
|:------------:|:--------------:|:---------------:|:-:|
|   baseline   |     81.40      |        -        | - |
|     0.05     |     80.74      |        -        | - |
|     0.10     |     81.66      |        -        | - |
|     0.15     |     81.69      |        -        | - |
|     0.20     | 81.47(anxiang) | 82.99(yongfeng) | - |
|     0.25     |     80.83      |        -        | - |
|     0.30     |     82.10      |        -        | - |
|     0.50     |       -        |        -        | - |


### random erasing repeat prob 0.2(online)

| erasing_prob |   1   |   2   |   3   |   4   |
|:------------:|:-----:|:-----:|:-----:|:-----:|
|     0.20     | 80.74 | 82.26 | 81.78 | 81.11 |


### 2019 11.12 baseline(dev-set)
train-dev set is split2

| experiment | rank-1 | mAP  | online |
|:----------:|:------:|:----:|:------:|
|     1      |  90.2  | 80.8 |  81.4  |
|     2      |  90.4  | 80.7 |   -    |
|     3      |  89.5  | 80.2 |   -    |
|  average   |  90.0  | 80.6 |   -    |

### 2019 11.13 
### `raranking k1=7,k2=3,lambda=0.85`
|     experiment      | rank-1(dev) | mAP(dev) | online | reranking |
|:-------------------:|:-----------:|:--------:|:------:|:---------:|
|      baseline       |    90.4     |   80.7   | 81.40  |   82.60   |
|       +sample       |    90.7     |   82.2   | 82.92  |   84.12   |
| +sample  +focalloss |    90.7     |   82.3   | 82.42  |   83.81   |
|  +sample  +bnneck   |    86.9     |   76.1   |   -    |     -     |

### 2019 11.14
### 分层学习率，FC部分 乘 10
| experiment  | rank-1 | mAP  | online | reranking |
|:-----------:|:------:|:----:|:------:|:---------:|
|  baseline   |  90.7  | 82.2 | 82.92  |   84.12   |
|    lr*10    |  91.6  | 83.6 | 83.62  |   84.67   |
| lr*10 fixbn |  91.3  | 83.0 | 82.60  |     -     |

### 2019 11.15
### sample 3x
### random erasing 0.2
### random sized rect crop

