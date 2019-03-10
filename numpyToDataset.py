import torch.utils.data
import numpy as np

trainX = 10 * np.random.random((10,5))
trainY = np.zeros((10,3))
trainY[:,1] = 1

## trainX : numpy.ndarray ( 10 * 5 )
## trainY : numpy.ndarray ( 10 * 3 )
print("raw train data")
print(trainX)
print(trainY)

train_ = torch.utils.data.TensorDataset(torch.from_numpy(trainX).float(), torch.from_numpy(trainY))
train_iter = torch.utils.data.DataLoader(train_, batch_size=5, shuffle=True)

##
for batch_idx,sample in enumerate(train_iter):
    print("BATCH-INDEX : ", batch_idx)
    print("BATCH : ", sample)