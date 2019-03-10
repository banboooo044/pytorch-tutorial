import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import numpy as np

class MultiLayerNet(nn.Module):
    '''
    全結合の多層ニューラルネットワーク
    '''
    def __init__(self,input_size,hidden_size,output_size,use_dropout = False,dropout_ration = 0.5):
        super(MultiLayerNet, self).__init__()
        # bias : default True
        self.layer1 = nn.Linear(input_size,hidden_size)
        self.layer2 = nn.Linear(hidden_size,output_size)
        self.use_dropout =  use_dropout
        self.dropout_ration = dropout_ration
        if self.use_dropout:
            self.dropout = nn.Dropout(p=self.dropout_ration)

    def forward(self,x):
        x = F.relu(self.layer1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.layer2(x)
        return F.softmax(x,dim=1)

    def accuracy(self, x , t):
        output = network(Variable(torch.from_numpy(x).float()))
        y_predicted = torch.argmax(output.data, dim=1).numpy()
        accuracy = (int)(100 * np.sum(y_predicted == t) / len(y_predicted))
        return accuracy



if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=False)
    network = MultiLayerNet(input_size=784, hidden_size=50, output_size=10,use_dropout=True)
    optimizer = optim.SGD(network.parameters(), lr=0.1,momentum=0.9)
    #optimizer = optim.Adam(network.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    train_ = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(t_train).long())
    train_iter = torch.utils.data.DataLoader(train_, batch_size=100, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    epoch_num = 10
    for epoch in range(epoch_num):
        for idx, data in enumerate(train_iter):
            optimizer.zero_grad()
            inputs, labels = data
            inputs, labels = Variable(inputs,requires_grad=True),Variable(labels)
            output = network(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    print('accuracy: {0}%'.format(network.accuracy(x_test,t_test)))

