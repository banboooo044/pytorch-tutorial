import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class MultiLayerNet(nn.Module):
    '''
    全結合の多層ニューラルネットワーク
    '''
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        super(MultiLayerNet, self).__init__()
        # bias : default True
        self.layer1 = nn.Linear(input_size,hidden_size)
        self.layer2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return F.softmax(x,dim=1)

    def accuracy(self, x , t):
        output = network(Variable(torch.from_numpy(x).float()))
        y_predicted = torch.argmax(output.data, dim=1).numpy()
        accuracy = (int)(100 * np.sum(y_predicted == t) / len(y_predicted))
        return accuracy


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=False)
    network = MultiLayerNet(input_size=784, hidden_size=50, output_size=10)
    optimizer = optim.SGD(network.parameters(), lr=0.1)

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    criterion = nn.CrossEntropyLoss()

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        optimizer.zero_grad()
        x_t = Variable(torch.from_numpy(x_batch).float(), requires_grad=True)
        t_t = Variable(torch.from_numpy(t_batch).long())
        output = network(x_t)
        loss = criterion(output, t_t)
        loss.backward()
        optimizer.step()

    print('accuracy: {0}%'.format(network.accuracy(x_test,t_test)))



