import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import sequence
import cv2
import os
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model_rnn=tf.keras.models.load_model(r'C:\Users\hp\Desktop\project assignments(DL)\notes_rnn.h5')
model_cnn=tf.keras.models.load_model(r'C:\Users\hp\Desktop\project assignments(DL)\cnn_vgg19_v3.h5')   #import the rnn and cnn model

def test_rx_image_for_Covid19(imagePath,new_model):
    img = cv2.imread(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    img = np.array(img) / 255.0

    pred = new_model.predict(img)
    pred_neg = round(pred[0][1] * 100)
    pred_pos = round(pred[0][0] * 100)
    if pred_pos>=80:
        output=1
    else:
        output=0
    return output

df=pd.read_csv(r'C:\Users\hp\Desktop\project assignments(DL)\covid-chestxray-dataset-master\metadata.csv')

df=df.drop_duplicates(['clinical_notes'])   #将医生诊断一栏有重复的全部删除
data=df.clinical_notes.values   #数据为医生诊断语句(array)
str_data=[]
for i in data:
    str_data.append(str(i))   #把数据存储为字符串形式
content=''.join(str_data)   #把数据字符串所有词合并
vocab_word=sorted(set(content.split()))   #把合并后的内容去除重复词并排序
char2idx = {u:i for i, u in enumerate(vocab_word)}    #将词和数字对应成字典

df=df[df.view=='PA']
df=df[df.modality=='X-ray']
df=sklearn.utils.shuffle(df)   #打乱data顺序，方便选择训练测试集
df_train=df.iloc[:196,:]
df_test=df.iloc[197:,:]

def vectorize_string(string):
    output=[]
    for sub in string:
        try:
            output.append(char2idx[sub])
        except:
            pass
    return np.array(output)

def data_label_idx(df):
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

    data_idx=[]
    for sub in str_data:
        data_idx.append(vectorize_string(sub.split()))   #用数字形式存储语句到列表中
    return label2num,data_idx

label2num_train,data_idx_train=data_label_idx(df_train)

image_path_list=[]
for i in range(len(df_train)):
    image_path_list.append(r"C:\Users\hp\Desktop\project assignments(DL)\covid-chestxray-dataset-master\images"+'\\'+df_train.iloc[i,:].filename)

pre_cnn_list=[]
for sub_path in image_path_list:
    pre_cnn_list.append(test_rx_image_for_Covid19(sub_path,model_cnn))

max_word=100   #把一个note填补或者截断为125的长度
X=sequence.pad_sequences(data_idx_train,maxlen=max_word)
pre_list_rnn=model_rnn.predict(X)

pre_list_rnn=np.squeeze(pre_list_rnn).tolist()


train_data=zip(pre_cnn_list,pre_list_rnn)
train_data_num=[]
for tup in train_data:
    train_data_num.append(list(tup))
train_data_num=np.array(train_data_num)
log_dir=os.path.join('logs_combine')
print(log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir,histogram_freq=1)

combine_network=Sequential()
combine_network.add(Dense(8,activation='relu'))
combine_network.add(Dense(16,activation='relu'))
combine_network.add(Dense(8,activation='relu'))
combine_network.add(Dense(1,activation='sigmoid'))
combine_network.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
combine_network.fit(train_data_num,label2num_train,batch_size=8,epochs=100,callbacks=[tensorboard_callback])
