import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import image

conv_layers=[
    Conv2D(64,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
    Conv2D(64,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    MaxPooling2D(pool_size=[2,2],strides=2,padding='same'),
    Conv2D(128,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    Conv2D(128,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    MaxPooling2D(pool_size=[2,2],strides=2,padding='same'),
    Conv2D(256,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    Conv2D(256,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    MaxPooling2D(pool_size=[2,2],strides=2,padding='same'),
    Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    MaxPooling2D(pool_size=[2,2],strides=2,padding='same'),
    Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    MaxPooling2D(pool_size=[2,2],strides=2,padding='same')
]

conv_net=Sequential(conv_layers)

fc_net=Sequential([
    Dense(256,activation=tf.nn.relu),
    Dense(128,activation=tf.nn.relu),
    Flatten(),
    ReLU(),
    Dense(1,activation=tf.nn.sigmoid)
])

full_net=Sequential([
    conv_net,
    fc_net
])

full_net.build(input_shape=(32,224,224,3))

full_net.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),loss=keras.losses.binary_crossentropy,metrics=['accuracy'])

train_data_generator=image.ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_data_generator=image.ImageDataGenerator(rescale=1./255)

train_generator=train_data_generator.flow_from_directory(
    r"C:\Users\hp\Desktop\project assignments(DL)\Dataset\Train",
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

validation_generator=test_data_generator.flow_from_directory(
    r"C:\Users\hp\Desktop\project assignments(DL)\Dataset\Val",
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

hist=full_net.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=2
)