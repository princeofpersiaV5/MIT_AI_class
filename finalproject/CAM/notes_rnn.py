import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import sequence
import os

def vectorize_string(string):
    output=[]
    for sub in string:
        try:
            output.append(char2idx[sub])
        except:
            pass
    return np.array(output)

df=pd.read_csv(r'C:\Users\hp\Desktop\project assignments(DL)\covid-chestxray-dataset-master\metadata.csv')   #读取metadata文件

df=df.drop_duplicates(['clinical_notes'])   #将医生诊断一栏有重复的全部删除
data=df.clinical_notes.values   #数据为医生诊断语句(array)
labels=df.finding.values   #标签为是否是新冠肺炎

label2num=[]
for i in labels:
    if i=='COVID-19':
        label2num.append(1)
    else:
        label2num.append(0)
label2num=np.array(label2num)
str_data=[]
for i in data:
    str_data.append(str(i))   #把数据存储为字符串形式
content=''.join(str_data)   #把数据字符串所有词合并
vocab_word=sorted(set(content.split()))   #把合并后的内容去除重复词并排序
char2idx = {u:i for i, u in enumerate(vocab_word)}    #将词和数字对应成字典
idx2char=np.array(vocab_word)

data_idx=[]
for sub in str_data:
    data_idx.append(vectorize_string(sub.split()))   #用数字形式存储语句到列表中

#把是新冠的标签存储为1，不是的存储为0
X_train,X_test,y_train,y_test=model_selection.train_test_split(data_idx,label2num,test_size=0.3,random_state=0,stratify=label2num)
X_val,X_test,y_val,y_test=model_selection.train_test_split(X_test,y_test,test_size=0.3,random_state=0,stratify=y_test)
#把数据按7:2:1的方式分割为训练集、验证集和测试集

max_word=100   #把一个note填补或者截断为125的长度
X_train=sequence.pad_sequences(X_train,maxlen=max_word)
X_val=sequence.pad_sequences(X_val,maxlen=max_word)
X_test=sequence.pad_sequences(X_test,maxlen=max_word)
vocabulary_size=6259
embeding_size=64
model=Sequential()
model.add(Embedding(vocabulary_size,embeding_size,input_length=max_word))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#建立神经网络模型
log_dir=os.path.join('logs')
print(log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir,histogram_freq=1)
batch_size=16
num_epochs=20
model.fit(X_train,y_train,validation_data=(X_val,y_val),batch_size=batch_size,epochs=num_epochs,callbacks=[tensorboard_callback])
