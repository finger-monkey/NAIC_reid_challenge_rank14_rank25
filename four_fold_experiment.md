# Experiment-Four-Fold

### Experiment baseline
```yml
INPUT:
  PIXEL_MEAN: [0.485, 0.456, 0.406]
  PIXEL_STD: [0.229, -0.224, -0.225]
SOLVER:
  OPTIMIZER_NAME: 'SGD'
  MAX_EPOCHS: 80
  BASE_LR: 0.005
  CENTER_LR: 0.5
  CENTER_LOSS_WEIGHT: 0.0005
  MARGIN: 1.0
  STEPS: [50, 70]
```
| experiment | split1(mAP/Rank-1) |   split2  |   split3  |   split4  |     avg    |
|:----------:|:------------------:|:---------:|:---------:|:---------:|:----------:|
|  baseline  |      84.7/92.7     | 85.2/93.0 | 83.6/93.3 | 80.1/91.2 | 83.4/92.55 |

### Experiment 001
```yml
INPUT:
  PIXEL_STD: [0.229, 0.224, 0.225]
```  
| experiment | split1(mAP/Rank-1) |   split2  |   split3  |   split4  |     avg    | result |
|:----------:|:------------------:|:---------:|:---------:|:---------:|:----------:|:------:|
|  baseline  |      84.7/92.7     | 85.2/93.0 | 83.6/93.3 | 80.1/91.2 | 83.4/92.55 |    -   |
|     001    |      83.7/92.7     | 83.5/89.9 | 83.5/92.5 | 80.0/92.6 | 82.7/91.92 |  worse |