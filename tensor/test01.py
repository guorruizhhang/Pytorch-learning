# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 10:59:19 2022

@author: 86186
"""

import numpy as np
import torch
data=np.array([1,2,3])
t1=torch.Tensor(data)
t2=torch.tensor(data,dtype=torch.float64)
t3=torch.as_tensor(data)
t4=torch.from_numpy(data)
data[0]=0
data[1]=0
data[2]=0
print(t1) #copy一个data作为数据来源
print(t2) #copy一个data作为数据来源
print(t3) #和data共享内存
print(t4) #和data共享内存
