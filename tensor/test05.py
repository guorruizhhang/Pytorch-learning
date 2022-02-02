# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 20:21:54 2022

@author: 86186
"""

#张量的缩减
import torch
import numpy as np
t1=torch.tensor([[1,2,3],
                 [4,5,6]],dtype=torch.float32)
t2=t1.sum()
t3=t1.numel()
t4=t1.sum().numel()
t5=t4<t1
print(t5)
#按照维度求和
t6=t1.sum(dim=0)
t7=t1.sum(dim=1)
print(t6) #t6=t1[0]+t1[1]+t1[2]，结果为为三个数组的和
print(t7) #t7列索引，分别对每个数组求和
#argmax函数，寻找最大值的索引位置
print(t1.max())
print(t1.argmax()) #结果是降维flatten后的索引
print(t1.flatten())
print(t1.max(dim=0)) #压缩为一行，索引为最大值的列索引
print(t1.max(dim=1)) #压缩为一列，索引为最大值的行索引
print(t1.mean(dim=0))

