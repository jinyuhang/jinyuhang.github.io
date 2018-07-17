# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 19:29:40 2018

@author: jjm
"""

import os
import numpy as np
import nltk
from gensim.models.word2vec import Word2Vec

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

def loaddata(file_path="data/docs/"):
    
    for  file_name in os.listdir(file_path):
        """
        读取数据文件
        """
        raw_text = ""
        if file_name.endswith(".txt"):
            with open(file_path+file_name)as fp:
                raw_text += fp.read() + "\n\n"
                
        return raw_text
    

raw_text = loaddata()
raw_text =raw_text.lower()

sentensor = nltk.data.load("tokenizers/punkt/english.pickle") # 加载nltk的句子模型
sents = sentensor.tokenize(raw_text) # 将语料切分成句子

corpus = [] # 存放语料
for sen in sents:
    corpus.append(nltk.word_tokenize(sen)) #将句子切分成词语

print(len(corpus)) # 输出句子的个数
print(corpus[:3]) # 查看前三个句子

w2v_model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4) #创建Word2Vector模型
raw_inputs = [item for sublist in corpus for item in sublist] #将整个语料转换成单词序列
text_stream = []
vocab = w2v_model.vocab #
for word in raw_inputs: #遍历输入语料中每个单词
    if word in vocab: # 如果单词在Word2Vector 中  则保留
        text_stream.append(word)
        

# 构建训练集
seq_length = 10
x = []
y = []
for i in range(0, len(text_stream) - seq_length):

    given = text_stream[i:i + seq_length] # 选择一段作为序列特征
    predict = text_stream[i + seq_length] # 选择下一个单词作为输出
    x.append(np.array([w2v_model[word] for word in given])) # 将特征转化为向量
    y.append(w2v_model[predict])


#model = Sequential()
#model.add(LSTM(256, dropout_W=0.2, dropout_U=0.2, input_shape=(seq_length, 128)))
#model.add(Dropout(0.2))
#model.add(Dense(128, activation='sigmoid'))
#model.compile(loss='mse', optimizer='adam')
#
#model.fit(x, y, nb_epoch=50, batch_size=4096)
#
#def predict_next(input_array):
#    x = np.reshape(input_array, (-1,seq_length,128))
#    y = model.predict(x)
#    return y
#
#def string_to_index(raw_input):
#    raw_input = raw_input.lower()
#    input_stream = nltk.word_tokenize(raw_input)
#    res = []
#    for word in input_stream[(len(input_stream)-seq_length):]:
#        res.append(w2v_model[word])
#    return res
#
#def y_to_word(y):
#    word = w2v_model.most_similar(positive=y, topn=1)
#    return word
#
#def generate_article(init, rounds=30):
#    in_string = init.lower()
#    for i in range(rounds):
#        n = y_to_word(predict_next(string_to_index(in_string)))
#        in_string += ' ' + n[0][0]
#    return in_string
#
#
#
#init = 'Language Models allow us to measure how likely a sentence is, which is an important for Machine'
#article = generate_article(init)
#print(article)






