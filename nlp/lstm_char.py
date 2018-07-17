# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:52:42 2018

使用LSTM 进行字母的生成

我们这里简单的文本预测就是，给了前置的字母以后，下一个字母是谁？
比如，Winsto, 给出 n Britai 给出 n

@author: jjm
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
#from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


def loaddata():
    """
    加载数据
    """
    raw_text = ""
    with open("data/Winston_Churchil.txt")  as fp:
        raw_text = fp.read()
        raw_text = raw_text.lower()
    return raw_text

def encode(raw_text):
    """
    对字符进行编码
    """
    chars = sorted(list(set(raw_text)))
    char2int = dict((c,i) for i,c in enumerate(chars))
    int2char = dict((i,c) for i,c in enumerate(chars))
    
    return char2int,int2char,chars

def prepare_data(raw_text,char2int,chars,seq_len = 100):
    """
    根据给定文本进行数据集的构建
    """
    # 构建基础的数据
    x = []
    y = []
    for i in range(0,len(raw_text) - seq_len):
        x_i = raw_text[i:i+seq_len]
        y_i = raw_text[i+seq_len]

        x.append([char2int[char] for char in x_i])
        y.append(char2int[y_i])
        
        
    #基于构建的样本数据转换成LSTM想要的格式
    num_sample = len(x)
    num_class = len(chars) # 类别的个数
    
    datax = np.reshape(x,(num_sample,seq_len,1))
    datay = np_utils.to_categorical(y)
    
    # 简单的归一化
    datax = datax / num_class
        
    return datax,datay

def build_model(datax,datay):
    model = Sequential()
    model.add(LSTM(128,input_shape=(datax.shape[1],datax.shape[2])))
    model.add(Dropout(0.2))
    
    model.add(Dense(datay.shape[1],activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="adam")
    
    return model

def predict(model,data_x,chars,seq_len=100):
    x = np.reshape(data_x,(1,seq_len,1))
    num_class = len(chars)
    x = x / num_class
    
    py = model.predict(x)
    
    return py

def string2index(raw_inputs,char2int,seq_len=100):
    res =[]
    for c in raw_inputs[len(raw_inputs) - seq_len]:
        res.append(char2int[c])
    
    return res

def y2char(y,int2char):
    max_index = y.argmax()
    char = int2char(max_index)
    return char

def generate_article(model,init,char2int,int2char,chars,rounds=5000):
    in_string = init.lower()
    for i in range(rounds):
        yhat = predict(model,string2index(in_string,char2int),chars)
        n = y2char(yhat)
        
        in_string += n
        
        return in_string

if __name__ == "__main__":
    raw_text = loaddata()
    char2int,int2char,chars = encode(raw_text)
    
    data_x,data_y = prepare_data(raw_text,char2int,chars)
    model = build_model(data_x,data_y)
    
    model.fit(data_x,data_y,nb_epoch=10,batch_size = 32)
    
    init = 'Professor Michael S. Hart is the originator of the Project'
    article = generate_article(model,init,char2int,int2char,chars)
    print(article)
#    print  data_x[:3]
    
    













