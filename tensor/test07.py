# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 16:43:04 2022

@author: 86186
"""

#数据加载器使用，数据训练
import  torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
train_set = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
    )
train_loader=torch.utils.data.DataLoader(train_set,batch_size=10)
'''print(len(train_set))   #图像个数
print(train_set.targets)#目标标签索引
print(train_set.targets.bincount())#每个标签类别的样本数/发生频率'''
sample=next(iter(train_set))
'''print(len(sample))
print(type(sample))'''
'''image=sample[0]  #像素张量
lable=torch.tensor(sample[1])
print(image.shape)
print(lable.shape)
plt.imshow(image.squeeze(),cmap='gray')#压缩为一维，进行绘图
print('lable',lable)'''
#批处理
batch=next(iter(train_loader))
images=batch[0]  #像素张量
lables=torch.tensor(batch[1])
print(images.shape)
print(lables.shape)
grid = torchvision.utils.make_grid(images, nrow=10)#创建网格，第一个参数为批处理个数，nrow表示每一行有几个网格
plt.figure(figsize=(15,15))
plt.imshow(np.transpose(grid,(1,2,0)),cmap='gray') #transpose将维度1,28,28]转换为[28,28,1]
print('lables',lables)  
'''imshow函数的参数为（imagesize,imagesize,channels）
image的格式为channels,imagesize,imagesize，使用transpoes转换数据格式'''
