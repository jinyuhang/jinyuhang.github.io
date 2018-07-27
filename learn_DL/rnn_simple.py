# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 09:39:44 2018
学习RNN
1 参考博客：https://zybuluo.com/hanbingtao/note/541458
2 参考代码：https://github.com/hanbt/learn_dl/blob/master/rnn.py

@author: jinjm1
"""

import numpy as np
from cnn import element_wise_op
from activators import ReluActivator
#from actovators import IdentityActivator

class RecurrentLayer(object):
    
    def __init__(self,input_width,state_width,activator,learning_rate):
        """
        初始化函数
        """
        self.input_width = input_width
        self.state_width = state_width
        self.activator = activator
        self.learning_rate = learning_rate
        
        self.times = 0
        self.state_list = [] # 每个元素为一个响亮
        self.state_list.append(np.zeros(state_width,1))
        self.U = np.random.uniform(-1e-4,1e-4,(state_width,input_width)) #初始化U矩阵
        self.W = np.random.uniform(-1e-4,1e-4,(state_width,state_width)) #初始化W矩阵
        
    def forward(self,input_aray):
        """
        进行前行计算
        """
        self.times += 1
        ux = np.dot(self.U,input_aray)
        ws = np.dot(self.W,self.state_list[-1])
        state = ux + ws
        element_wise_op(state,self.activator.forward)
        self.state_list.append(state)
        
    def backward(self,sensitivity_array,activator):
        """
        实现BPTT算法: 循环层的误差传播
        """
        self.calc_delta(sensitivity_array,activator)
        self.calc_gradient()
        
    def update(self):
        """
        按照梯度下降，更新权重
        """
        self.W -= self.learning_rate * self.gradient
        
    def calc_delta(self,sensitivity_array,activator):
        """
        计算梯度
        """
        self.delta_list = [] # 用来保存各个时刻的误差项
        for i in range(self.times):
            self.delta_list.append()
    
    
    def calc_delta_k(self,k,activator):
        """
        根据k+1 时刻的delta计算k时刻的delta
        """
        state = self.state_list[k+1].copy()
        element_wise_op(self.state_list[k+1],activator.backward)
        self.delta_list[k] = np.dot(
                np.dot(self.delta_list[k+1].T,self.W),
                np.diag(state[:,0])
                ).T
    
    def calc_gradient(self):
        """
        保存各个时刻的权重梯度
        """
        self.gradient_list = []
        for t in range(self.times+1):
            gra = np.zeros((self.state_width,self.state_width))
            self.gradient_list.append(gra)
            
        for t in range(self.times,0,-1):
            self.calc_gradient_t(t)
            
        # 实际的梯度是各个时刻梯度之和
        self.gradient = reduce(lambda a,b:a+b,
                               self.gradient_list,
                               self.gradient_list[0]) # [0]被初始化为0且没有被修改过
    
    def calc_gradient_t(self,t):
        """
        计算每个时刻t权重的梯度
        """
        
        gradient = np.dot(self.delta_list[t],self.state_list[t-1].T)
        
        self.gradient_list[t] = gradient
        
    def reset_state(self):
        """
        重置状态
        """
        self.times = 0 #将当前时刻初始化为0
        self.state_list = [] # 保存各个时刻的state
        self.state_list.append(
                np.zeros((self.state_width,1)) #初始化s0
                )
        
def data_set():
    x = [np.array([[1],[2],[3]]),
         np.array([[2],[3],[4]])
         ]
    d = np.array([[1],[2]])
    
    return x,d

def test():
    x,d = data_set()
    model = RecurrentLayer(3,2,ReluActivator(),1e-3)
    model.forward(x[0])
    model.forward(x[1])
    model.backward(d,ReluActivator())
    
    return model,x,d

if __name__ == "__main__":
    
    x,d = data_set()
    model,x,d = test()
            
            
    
    
    
    
    
    
    
    
    
    
    