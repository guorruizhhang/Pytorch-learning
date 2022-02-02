# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:01:58 2022

@author: 86186
"""
#图像识别为例，张量的压缩
import torch
t1=torch.tensor([[1,1,1,1],
                 [1,1,1,1],
                 [1,1,1,1],
                 [1,1,1,1]])#图片1

t2=torch.tensor([[2,2,2,2],
                 [2,2,2,2],
                 [2,2,2,2],
                 [2,2,2,2]])#图片2
                 
t3=torch.tensor([[3,3,3,3],
                 [3,3,3,3],
                 [3,3,3,3],
                 [3,3,3,3]])#图片3
t=torch.stack((t1,t2,t3))
#print(t)
#print(t.size()) #[3,4,4] 3，批次的大小14，4表示高和宽
t4=t.reshape(3,1,4,4)     #1表示选定图片后的单一通道
print(t4)  
t5=t.flatten(start_dim=1) #flatten是自带函数，从第二个索引开始压缩
print(t5)
        