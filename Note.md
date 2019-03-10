# Multi-layer-Net

## MNIST
### 実験条件
epoch = 10 , batch size = 100
| optimizer | optimizerParam | accuracy |
|:------------:|:-------:|:--------:|
| momentumSGD | lr = 0.1  | 93% |
| momentumSGD | lr = 0.1, momentum = 0.9 | 96% |
| momentumSGD | lr = 0.1, weight_decay = 1e-4 | 92% |
| Adam | lr=0.001, betas=(0.9, 0.999) | 95% |
| momentumSGD | lr = 0.1, momentum = 0.9, drop_out_rate = 0.5 | 90% |
| momentumSGD | lr = 0.1, momentum = 0.9, batch_normarized | 94% |
