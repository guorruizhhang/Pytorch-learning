# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:41:23 2022

@author: 86186
"""

import torch
t=torch.tensor([[1,1,1,1],
               [2,2,2,2],
               [3,3,3,3]],dtype=torch.float32)
print(len(t.shape))
print(t.numel())
print(t.reshape(4,3)) #重塑
print(t.reshape(2,2,3)) #重塑

t1=t.reshape(1,12) #仅reshape
def flatten(t):
    t=t.reshape(1,-1)
    t=t.squeeze()
    return t
t2=flatten(t)
print(t1)
print(t2) #f

               