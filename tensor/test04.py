# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:07:37 2022

@author: 86186
"""

#张量运算
import torch
import numpy as np
t1=torch.tensor([[1,2],
                 [3,4]],dtype=torch.float32)
t2=torch.tensor([[6,7],
                 [8,9]],dtype=torch.float32)
t3=t1+t2
t4=t1+2
t5=t1-2
t6=t1*2
t7=t1/2
t8=t1.add(2) #先进行标量广播，再进行张量运算
t9=t1.sub(2)
t10=t1.mul(2)
t11=t1.div(2)
a=np.broadcast_to(2,t1.shape) #广播函数，复制为高纬度的张量
print(a)
b=torch.tensor([2,4])
c=t1+b
print(c)
#比较运算
#torch.eq/ge/gt/lt/le表示equal,greater than,greater,less than,less equal