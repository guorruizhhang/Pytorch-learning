# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 16:19:40 2022

@author: 86186
"""

#下载fashion数据集
import  torch
import torchvision
import torchvision.transforms as transforms
train_set = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
    )