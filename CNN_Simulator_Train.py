#-*- coding: utf-8 -*-
"""
Created on Sat Feb 25 16:53:03 2017

@author: admin
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from keras.models import load_model
from keras.optimizers import Adam
import json
from PIL import Image
import os
import cv2
import numpy as np
import glob
import pandas as pd
import io
 
path1="H:\Test\project\cropped & resized dataset\center_cropped_ROI"

listing = os.listdir(path1)
immatrix = np.array([np.array(cv2.imread(path1 + '\\' + 'center_'+str(i+1)+'.png')).flatten() for i in range(0,1971)] ,'f')           


a=[]
df = pd.read_csv('H:\Test\project\driving_log.csv', header=None, usecols=[3,])
df=df.values
#print df[3]
for i in range(0,len(df)):
    #print i
    a.append(df[i][0])
a=np.array(a)
a=a.flatten()
print (a)
#data,label = shuffle(immatrix,label, random_state=2)
train_data=[immatrix,a]

(x, y) = (train_data[0],train_data[1])
#
print(x)
##split X and y into training and testing sets
#
X_train, X_test, y_train, y_test= train_test_split(x, y,test_size=0.09,random_state=4)
#
##
X_train = X_train.reshape(X_train.shape[0], 3, 28,28)
X_test = X_test.reshape(X_test.shape[0], 3, 28,28)

##
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
##
X_train /= 255
X_test /= 255

##
print('X_train shape:', X_train.shape)
print('Y_train shape:', y_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
#print('CNN3')
#Y_train = np_utils.to_categorical(y_train,3)
#Y_test = np_utils.to_categorical(y_test,3)
#
#
#print (y_train[28])
#
model = Sequential()
## input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
## this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(96,5,5, border_mode='valid', input_shape=(3,28,28)))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Convolution2D(32,3,3))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Convolution2D(16,2,2))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Convolution2D(8,2,2))
model.add(Activation('tanh'))
#
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
#
model.add(Flatten())
## Note: Keras does automatic shape inference.
model.add(Dense(130))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#
model.add(Dense(1))
#
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])#optimizer sgd
#
model.fit(X_train, y_train, batch_size=32, nb_epoch=10, verbose=1, validation_data=(X_test, y_test),shuffle=True)
score = model.predict(X_test, y_test , verbose=0)
#
#print('Test accuracy:', score[1])
print(model.predict(X_test[20:30]))
print(y_test[20:30])
model.save(' Simulator_CNN_1.h5')
print("Saved")
#
#print(score)