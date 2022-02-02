# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 12:53:16 2022

@author: 86186
"""

#定义CNN
import torch.nn as nn
class networks(nn.Module):
    def __init__(self):
        super(networks, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2=nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1=nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2=nn.Linear(in_features=120, out_features=60)
        self.fc2=nn.Linear(in_features=60, out_features=10)
     
    