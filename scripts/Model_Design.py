#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:31:32 2019

@author: andy
"""

from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.convolutional import Conv3D
from keras.initializers import he_normal
from keras.initializers import Orthogonal
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten, Reshape, Permute
from keras.layers.merge import Concatenate
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.layers.convolutional import Cropping3D
from keras.layers import UpSampling3D
from keras.layers import concatenate
from keras.layers.advanced_activations import PReLU
from keras.layers import LeakyReLU
from keras.utils import print_summary
from keras import regularizers
from keras.optimizers import RMSprop
import keras.backend as K
from keras.optimizers import Adam
import numpy as np
from keras.activations import softmax
from keras.engine import InputLayer

#------------------------------------------------------------------------------------------


def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def Generalised_dice_coef_multilabel2(y_true, y_pred, numLabels=2):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return numLabels + dice

def dice_coef_multilabel6(y_true, y_pred, numLabels=6):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return numLabels + dice
def w_dice_coef(y_true, y_pred, PENALTY):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f) * PENALTY
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multilabel0(y_true, y_pred):
    index = 0
    dice = dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice
def dice_coef_multilabel1(y_true, y_pred):
    index = 1
    dice = dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice
def dice_coef_multilabel2(y_true, y_pred):
    index = 2
    dice = dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice



dpatch = [13,75,75]
output_classes = 2
conv_features_downsample = [30,30,30,30,30,30,30,30,30]
conv_features = [20,20,20,20,30,30,30,30,30,30,30,50,50,50,50] #[50,50,50,50,50,50,50,70,70,70,70,100,100]
fc_features = [60,60,80,100]#[50,50,50,50]
num_channels = 3
L2 = 1
dropout = [0]
learning_rate = 0.1


data_input = Input((None,None,None, num_channels))

data_input = Input((13,75,75, num_channels))
x = AveragePooling3D((1,3,3), padding='valid', strides=(1,1,1))(data_input)


x        = Conv3D(filters = 20, 
                       kernel_size = (1,3,3),
                       dilation_rate=(1,5,5),
                       #kernel_initializer=he_normal(seed=seed),
                       kernel_initializer=Orthogonal(),
                       kernel_regularizer=regularizers.l2(L2))(data_input)
x        = BatchNormalization()(x)
x        = Activation('relu')(x)

for feature in conv_features[0:5]:    # reduce all dimensions in -24
    x        = Conv3D(filters = feature, 
                       kernel_size = (3,3,3), 
                       #kernel_initializer=he_normal(seed=seed),
                       kernel_initializer=Orthogonal(),
                       kernel_regularizer=regularizers.l2(L2))(x)
    x        = BatchNormalization()(x)
    x        = Activation('relu')(x)

for feature in (conv_features[0:9]):    # reduce in -36
    x        = Conv3D(filters = feature, 
                       kernel_size = (1,5,5), 
                       #kernel_initializer=he_normal(seed=seed),
                       kernel_initializer=Orthogonal(),
                       kernel_regularizer=regularizers.l2(L2))(x)
    x        = BatchNormalization()(x)
    x        = Activation('relu')(x)

for feature in (conv_features[0:9]):    
    x        = Conv3D(filters = feature, 
                       kernel_size = (1,3,3), 
                       #kernel_initializer=he_normal(seed=seed),
                       kernel_initializer=Orthogonal(),
                       kernel_regularizer=regularizers.l2(L2))(x)
    x        = BatchNormalization()(x)
    x        = Activation('relu')(x)

for feature in (conv_features[8:10]):  
    x        = Conv3D(filters = feature, 
                       kernel_size = (1,1,1), 
                       #kernel_initializer=he_normal(seed=seed),
                       kernel_initializer=Orthogonal(),
                       kernel_regularizer=regularizers.l2(L2))(x)
    #x        = BatchNormalization()(x)
    #x        = Activation('relu')(x)
    x        = LeakyReLU()(x)
x        = BatchNormalization()(x)
x        = Dropout(rate = dropout[0])(x)
   
Y_coords = Input((None,None,None,1))
Z_coords = Input((None,None,None,1))

#coords = concatenate([Y_coords, Z_coords])  

Y = Conv3D(filters = 2, 
           kernel_size = (1,1,1), 
           activation='sigmoid',
           name = 'Y_Processing',
           weights = [np.array([[[[[0.5,  0.5 ]]]]]), np.array([0., 1.])],
           kernel_initializer=Orthogonal(),
           kernel_regularizer=regularizers.l2(L2))(Y_coords)
       
Z = Conv3D(filters = 2, 
           kernel_size = (1,1,1), 
           activation='sigmoid',
           name = 'Z_Processing',
           weights = [np.array([[[[[0.5,  0.5 ]]]]]), np.array([0., 1.])],
           kernel_initializer=Orthogonal(),
           kernel_regularizer=regularizers.l2(L2))(Z_coords)  

coords = concatenate([Y, Z])   

coords = Conv3D(filters = 1, 
                kernel_size = (1,1,1), 
                activation='sigmoid',
                kernel_initializer=Orthogonal(),
                kernel_regularizer=regularizers.l2(L2))(coords)  
        
x = concatenate([x, coords])    

x        = Conv3D(filters = output_classes, 
           kernel_size = (1,1,1), 
           #kernel_initializer=he_normal(seed=seed),
           kernel_initializer=Orthogonal(),
           kernel_regularizer=regularizers.l2(L2))(x)
x        = Activation(softmax)(x)

model     = Model(inputs=[data_input, Y_coords, Z_coords], outputs=x)

model.summary()

model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=learning_rate), metrics=[dice_coef_multilabel0,dice_coef_multilabel1])
                   