import os
import datetime
import random
import tensorflow as tf
import statistics as stat
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import seaborn as sns
from imutils import  paths

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
def load_image_folder_direct(input_dataset_path,output_dataset_path,img_num_select):
    img_number=0
    pathlist=Path(input_dataset_path).glob('**/*.*')
    nof_samples=img_num_select
    rc=[]
    for k,path in enumerate(pathlist):
        if k<nof_samples:
            rc.append(str(path))
            shutil.copy2(path,output_dataset_path)
            img_number+=1
        else:
            i=random.randint(0,k)
            if i<nof_samples:
                rc[i]=str(path)
    print('{} selected Images on folder {}:'.format(img_number,
                                                    output_dataset_path))

def ceildiv(a,b):
    return -(-a//b)

def plot_from_files(imspaths,figsize=(10,5),rows=1,titles=None,maintitle=None):
    f=plt.figure(figsize=figsize)
    if maintitle is not None:
        plt.suptitle(maintitle,fontsize=10)
    for i in range(len(imspaths)):
        sp=f.add_subplot(rows,ceildiv(len(imspaths),rows),i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i],fontsize=16)
        img=plt.imread(imspaths[i])
        plt.imshow(img)

def test_rx_image_for_Covid19(imagePath,new_model):
    img = cv2.imread(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    img = np.array(img) / 255.0

    pred = new_model.predict(img)
    pred_neg = round(pred[0][1] * 100)
    pred_pos = round(pred[0][0] * 100)

    if np.argmax(pred, axis=1)[0] == 1:
        plt.title(
            '\nPrediction: [NEGATIVE] with prob: {}% \nNo Covid-19\n'.format(
                pred_neg),
            fontsize=12)
    else:
        plt.title(
            '\nPrediction: [POSITIVE] with prob: {}% \nPneumonia by Covid-19 Detected\n'
            .format(pred_pos),
            fontsize=12)

    img_out = plt.imread(imagePath)
    plt.imshow(img_out)
    return pred_pos


def test_rx_image_for_Covid19_2(imagePath, neg_cnt, pos_cnt,new_model):
    img = cv2.imread(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    img = np.array(img) / 255.0

    pred = new_model.predict(img)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    # print(np.argmax(pred, axis=1))

    if np.argmax(pred, axis=1)[0] == 1:
        neg_cnt += 1
    else:
        pos_cnt += 1

    return pred[0][0], neg_cnt, pos_cnt

def test_rx_image_for_Covid19_batch(img_lst):
    neg_cnt = 0
    pos_cnt = 0
    predictions_score = []
    for img in img_lst:
        pred, neg_cnt, pos_cnt = test_rx_image_for_Covid19_2(img, neg_cnt, pos_cnt)
        predictions_score.append(pred)
    print ('{} positive detected in a total of {} images'.format(pos_cnt, (pos_cnt+neg_cnt)))
    return  predictions_score, neg_cnt, pos_cnt

def plot_prediction_distribution(dist, name='Dataset Prediction Score Distribution'):
    f, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    sns.despine(left=True)
    sns.distplot(dist, hist=True, color = 'royalblue', ax=axes[0])
    sns.boxplot(dist, color = 'firebrick', ax=axes[1])
    plt.suptitle(name, size = 20)
    plt.setp(axes, yticks=[]);

def plot_accuraccy(history):
    plt.style.use('seaborn-paper')
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

INIT_LR=0.0001
EPOCHS=30
BS=16
NODES_DENSE0=128
DROPOUT=0.5
MAXPOOL_SIZE=(2,2)
ROTATION_DEG=15
SPLIT=0.2

image_paths=list(paths.list_images(r"C:\Users\hp\Desktop\project assignments(DL)\Dataset\Train"))

data=[]
labels=[]
for image_path in image_paths:
    label=image_path.split(os.path.sep)[-2]
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image=cv2.resize(image,(224,224))
    data.append(image)
    labels.append(label)

data=np.array(data)/255.
labels=np.array(labels)

lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=SPLIT,stratify=labels,random_state=0)
trainAug=ImageDataGenerator(rotation_range=ROTATION_DEG,fill_mode="nearest",shear_range=20,zoom_range=0.2,horizontal_flip=True)

baseModel = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=MAXPOOL_SIZE)(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(NODES_DENSE0, activation="relu")(headModel)
headModel = Dropout(DROPOUT)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
    layer.trainable = False

opt=Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
log_dir=os.path.join('logs_cnn')
print(log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir,histogram_freq=1)
H=model.fit(
    trainAug.flow(trainX,trainY,batch_size=BS),
    steps_per_epoch=len(trainX)//BS,
    validation_data=(trainX,trainY),
    validation_steps=len(testX)//BS,
    epochs=EPOCHS,
    callbacks=[tensorboard_callback]
)

plot_accuraccy(H)
