# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 18:57:28 2022

@author: 86186
"""

#类的定义，使用
class lizard:
    def __init__(self,name): #构造函数，创建一个类的实例
        self.name = name
    def set_name(self,name): #方法，函数，此处为改变实例的名称
        self.name = name
lizard1=lizard('deep')
print(lizard1.name)
lizard1.set_name('learning')
print(lizard1.name)