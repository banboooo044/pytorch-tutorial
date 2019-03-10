# Multi-layer-Net

### 初期設定

| epoch | batch size | optimizer | optimizerParam | accuracy |
|:-----------:|:------------:|:------------:|:-------:|:--------:|
| 10 | 100 | momentumSGD | lr = 0.1  | 93% |
| 10 | 100 | momentumSGD | lr = 0.1, momentum = 0.9 | 96% |
| 10 | 100 | momentumSGD | lr = 0.1, weight_decay = 1e-4 | 92% |
| 10 | 100 | Adam | lr=0.001, betas=(0.9, 0.999) | 95% |
| 10 | 100 | momentumSGD | lr = 0.1, momentum = 0.9, drop_out_rate = 0.5 | 90% |

結果 : accuracy 93%

