# -*- coding: utf-8 -*-
"""
@author: owner
"""
from __future__ import print_function
from __future__ import absolute_import
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import numpy as np
import theano
import glob 
path1="H:\Test\FastBurstCamera\Grayscaled"
from keras import backend as K


listing = os.listdir(path1)
immatrix = np.array([np.array(cv2.imread(path1 + '\\' + 'sample_'+str(i)+'.jpg',0)).flatten() for i in range(0,800)] ,'f')           
label=np.ones((800,),dtype=int)
label[0:223]=1 #left
label[223:632]=2#right
label[632:801]=3#straight
#data,label = shuffle(immatrix,label, random_state=2)
train_data=[immatrix,label]
(x, y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=4)


X_train = X_train.reshape(X_train.shape[0],1, 28,28)
X_test = X_test.reshape(X_test.shape[0],1, 28,28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255
print(len(X_test[0]))
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print('CNN3')
Y_train = np_utils.to_categorical(y_train,4)
print ("hello")
Y_test = np_utils.to_categorical(y_test,4)

model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 32 convolution filters of size 3x3 each.

model.add(Convolution2D(32,3,3, border_mode='valid', input_shape=(1,28,28),name='conv_1'))

convout1=Activation('relu')
model.add(convout1)
model.add(Convolution2D(64,3,3,name='conv_2'))
convout2=Activation('relu')
model.add(convout2)
model.add(Convolution2D(96,3,3,name='conv_3'))
convout3=Activation('relu')
model.add(convout3)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(130))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(4))
model.add(Activation('softmax'))
model.load_weights('H:\Test\CNN8.h5')
print(model.predict_classes(X_test[40:41]))
print(Y_test[40:50])



from keras import backend as K

import pylab as pl
import matplotlib.cm as cm
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import pylab as pl
import matplotlib.cm as cm
# K.learning_phase() is a flag that indicates if the network is in training or
# predict phase. It allow layer (e.g. Dropout) to only be applied during training
inputs = [K.learning_phase()] + model.inputs 

_convout1_f = K.function(inputs, [convout1.output])
def convout1_f(X):
    # The [0] is to disable the training phase flag
    return _convout1_f([0] + [X])
    
# utility functions
from mpl_toolkits.axes_grid1 import make_axes_locatable

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)

# Visualize convolution result (after activation)

import numpy.ma as ma
def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

#Visualize weights
i=40
X = X_test[i:i+1]

W = model.layers[0].W.get_value(borrow=True)
W = np.squeeze(W)
print("W shape : ", W.shape)

pl.figure(figsize=(15,15))
pl.title('conv1 weights')
nice_imshow(pl.gca(), make_mosaic(W,6,6), cmap=cm.binary)

pl.figure()
pl.title('input')
nice_imshow(pl.gca(), np.squeeze(X), vmin=0, vmax=1, cmap=cm.binary)

print(Y_test[i:i+1])
#
#
#
#


#
#i = 11
#
## Visualize the first layer of convolutions on an input image
#X = X_test[i:i+1]
#C1 = convout1_f(X)
#C1 = np.squeeze(C1)
#print("C1 shape : ", C1.shape)
#
#pl.figure(figsize=(15,15))
#pl.suptitle('convout1')
#nice_imshow(pl.gca(), make_mosaic(C1,10,10), cmap=cm.binary)