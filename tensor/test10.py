# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 15:12:49 2022

@author: gr_zhang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)  # 这里并不是必须的，默认情况下是打开的

print(torch.__version__)
print(torchvision.__version__)
# 一、训练数据获取
train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    )
# 二、创建网络
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
    
    def forward(self, t):
        # Input Layer
        t = t
        # Conv1
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # Conv2
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # FC1
        t = t.reshape(-1, 12*4*4)
        t = F.relu(self.fc1(t))
        # FC2
        t = F.relu(self.fc2(t))
        # Output
        t = self.out(t)
        return t
# 调用network实例
network = Network()
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
batch = next(iter(train_loader))
images, labels = batch
# 计算损失
preds = network(images)
loss = F.cross_entropy(preds,labels)
loss.item()   #获得损失的值

# 计算损失的梯度
loss.backward()      #反向传播
# 更新权重
optimizer = optim.Adam(network.parameters(), lr =0.01)
loss.item()      # 显示当前loss值

# 定义函数用于计算预测正确的数目
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
print(loss.item())
print(get_num_correct(preds, labels))
# 更新权重
optimizer.step()

preds = network(images)
loss = F.cross_entropy(preds,labels)
print(loss.item())
print(get_num_correct(preds, labels))