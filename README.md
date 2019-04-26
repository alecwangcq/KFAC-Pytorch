# K-FAC_pytorch
Pytorch implementation of [K-FAC](https://arxiv.org/abs/1503.05671) and [E-KFAC](https://arxiv.org/abs/1806.03884). (Only support single-GPU training, need modifications for multi-GPU.)
## Requiresments
```
pytorch 0.4.0
python 3.6.0
```
## How to run
```
python main.py --dataset cifar10 --optimizer kfac --network vgg16_bn  --epoch 100 --milestone 40,80 --learning_rate 0.01 --damping 0.03 --weight_decay 0.003
```


## Performance
For K-FAC and E-KFAC, the search range of learning rates, weight decay and dampings are:<br>
(1) learning rate = [3e-2, 1e-2, 3e-3] <br>
(2) weight decay = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4] <br>
(3) damping = [3e-2, 1e-3, 3e-3]

For SGD: <br>
(1) learning rate = [3e-1, 1e-1, 3e-2] <br>
(2) weight decay = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

#### CIFAR10

| Optimizer | Model                              | Acc.        | learning rate | weight decay |  damping |
|---------- | ---------------------------------- | ----------- | ------------- | -------------| ----------- |
| KFAC   | [VGG16_BN](https://arxiv.org/abs/1409.1556)  | 93.86% | 0.01 | 0.003 | 0.03 |
| E-KFAC | [VGG16_BN](https://arxiv.org/abs/1409.1556)  | 94.00% | 0.003 | 0.01 | 0.03 |
| SGD    | [VGG16_BN](https://arxiv.org/abs/1409.1556)  | 94.03% | 0.03 | 0.001 | - |
| KFAC   | [ResNet110](https://arxiv.org/abs/1512.03385)| 93.59% | 0.01 | 0.003 | 0.03 |
| E-KFAC | [ResNet110](https://arxiv.org/abs/1512.03385)| 93.37% | 0.003 | 0.01 | 0.03 |
| SGD    | [ResNet110](https://arxiv.org/abs/1512.03385)| 94.14% | 0.03 | 0.001 | - |



#### CIFAR100

| Optimizer | Model                              | Acc.        | learning rate | weight decay |  damping |
|---------- | ---------------------------------- | ----------- | ------------- | -------------| ----------- |
| KFAC   | [VGG16_BN](https://arxiv.org/abs/1409.1556)  | 74.09% | 0.003 | 0.01 | 0.03 |
| E-KFAC | [VGG16_BN](https://arxiv.org/abs/1409.1556)  | 73.20% | 0.01 | 0.01 | 0.03 |
| SGD    | [VGG16_BN](https://arxiv.org/abs/1409.1556)  | 74.56% | 0.03 | 0.003 | - |
| KFAC   | [ResNet110](https://arxiv.org/abs/1512.03385)| 72.71% | 0.003 | 0.01 | 0.003 |
| E-KFAC | [ResNet110](https://arxiv.org/abs/1512.03385)| 72.32% | 0.03 | 0.001 | 0.03 |
| SGD    | [ResNet110](https://arxiv.org/abs/1512.03385)| 72.60% | 0.1 | 0.0003 | - |

## Others
If you have any questions or suggestions, please feel free to contact me via alecwangcq at gmail , com!
