# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:30:56 2019

@author: hirsch
"""

from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.convolutional import Conv3D
from keras.initializers import he_normal
from keras.initializers import Orthogonal, RandomNormal, RandomUniform
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
from keras_radam import RAdam
from keras.layers import Add
from keras.utils import multi_gpu_model
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#tf.keras.backend.set_session(tf.Session(config=config))
#------------------------------------------------------------------------------------------
from keras.activations import relu 

def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f**2) + K.sum(y_pred_f**2) + smooth)

def Generalised_dice_coef_multilabel2(y_true, y_pred, numLabels=2):
    """This is the loss function to MINIMIZE. A perfect overlap returns 0. Total disagreement returns numeLabels"""
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return numLabels + dice

def dice_coef_relu(y_true, y_pred):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(2. * relu(y_true_f,threshold=0.5)) + K.sum(y_pred_f) + smooth)

def Generalised_dice_coef_multilabel2_ReLU(y_true, y_pred, numLabels=2):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef_relu(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
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

def dice_coef_multilabel_bin0(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,:,0], K.round(y_pred[:,:,:,:,0]))
    return dice

def dice_coef_multilabel_bin1(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,:,1], K.round(y_pred[:,:,:,:,1]))
    return dice



class ToyModel3D():
    
    def __init__(self, loss_function, use_softmax):
        self.loss = loss_function
        self.use_softmax = use_softmax
    
    def createModel(self):
    
        x_in = Input((9,9,9,1))
        x = x_in
        for _ in range(4):    # reduce all dimensions in -12
            x        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               kernel_initializer=Orthogonal())(x)
            x        = LeakyReLU()(x)                              
            x        = BatchNormalization()(x)   

        x        = Conv3D(filters = 2, 
                          kernel_size = (1,1,1), 
                          kernel_initializer=Orthogonal())(x)
        if self.use_softmax:
            y        = Activation('softmax')(x)
        else:
            y        = Activation('sigmoid')(x)
           
        model     = Model(inputs=[x_in], outputs=y)
        
        if self.loss == 'crossentropy':
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        elif self.loss == 'dice':    
            model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=0.001), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])

        #model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=0.001), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])
        return model


class ToyModel3D_B():
    
    def __init__(self, loss_function, use_softmax, n_filters, L2):
        self.loss = loss_function
        self.use_softmax = use_softmax
        self.n_filters = n_filters
        self.L2 = L2
    
    def createModel(self):
    
        x_in = Input((9,9,9,1))
        x = x_in
        for _ in range(4):    # reduce all dimensions in -12
            x        = Conv3D(filters = self.n_filters, 
                               kernel_size = (3,3,3), 
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x)
            x        = LeakyReLU()(x)                              
            x        = BatchNormalization()(x)   


        if self.use_softmax:
            x        = Conv3D(filters = 2, 
                              kernel_size = (1,1,1), 
                              kernel_initializer=Orthogonal())(x)            
            y        = Activation('softmax')(x)
        else:
            x        = Conv3D(filters = 2, 
                              kernel_size = (1,1,1), 
                              kernel_initializer=Orthogonal())(x)                   
            y        = Activation('sigmoid')(x)
           
        model     = Model(inputs=[x_in], outputs=y)
        
        if self.loss == 'crossentropy':
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        elif self.loss == 'dice':    
            model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=0.001), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])

        #model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=0.001), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])
        return model

class ToyModel():
    def __init__(self, loss_function, use_softmax):
        self.loss = loss_function
        self.use_softmax = use_softmax

    def createModel(self):
        x_in = Input((32,))
        x = Dense(100)(x_in)
        x= Activation('relu')(x)
        if self.use_softmax:
            x = Dense(2)(x)
            y = Activation('softmax')(x)
        else:
            x = Dense(2)(x)
            y = Activation('sigmoid')(x)
        model     = Model(inputs=[x_in], outputs=y)
        if self.loss == 'crossentropy':
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])
        elif self.loss == 'dice':    
            model.compile(loss=Generalised_dice_coef_multilabel2_general, optimizer=Adam(lr=0.001), metrics=['acc'])
        return model


class MultiPriors_v0():
    
    def __init__(self, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay, loss_function):
        
        self.output_classes = output_classes
        self.conv_features = [20,20,20,20,30,30,30,30,30,30,30,50,50,50,50] #[50,50,50,50,50,50,50,70,70,70,70,100,100]
        self.fc_features = [60,60,80,100]#[50,50,50,50]
        self.num_channels = num_channels
        self.L2 = L2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_decay = optimizer_decay
        self.loss_function = loss_function
    
    def createModel(self):
    
        Context = Input((None,None,None,1), name = 'Context')
        T1post = Input((None,None,None, 1),name = 'T1post_input')
        Sub = Input((None,None,None, 1),name = 'Sub_input')
       
        ########################  T1 post pathway #########################
        #############   High res pathway   ##################         
        x11        = Cropping3D(cropping = ((0,0),(20,20),(20,20)), input_shape=(None,None,None, self.num_channels),name = 'T1post_Detail')(T1post)
        
        # reduced original input by -40    : 13,35,35     
   
            
        for feature in self.conv_features[0:6]:    # reduce all dimensions in -12
            x11        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x11)
            x11        = LeakyReLU()(x11)                              
            x11        = BatchNormalization()(x11)   
 
        for feature in (self.conv_features[0:7]):    # reduce in -36
            x11        = Conv3D(filters = 40, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x11)
            x11        = LeakyReLU()(x11)                              
            x11        = BatchNormalization()(x11)                
            
        x12 = Context #AveragePooling3D(pool_size=(1, 3, 3), name='T1post_Context')(T1post)    
        # (13, 25, 25)        
        for iii in range(6):    
            x12        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               name='T1post_Context_{}'.format(iii),
                               kernel_regularizer=regularizers.l2(self.L2))(x12)
            x12        = LeakyReLU()(x12)                              
            x12        = BatchNormalization()(x12) 
 
        for jjj in range(5):    
            x12        = Conv3D(filters = 40, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               name='T1post_Context_{}'.format(6+jjj),
                               kernel_regularizer=regularizers.l2(self.L2))(x12)
            x12        = LeakyReLU()(x12)                              
            x12        = BatchNormalization()(x12)     
     
        x12   =  UpSampling3D(size=(1,3,3))(x12)
        # Result: (1,13,13)

        x1 = concatenate([x11,x12])

        ########################  Sub pathway #########################
        #############   High res pathway   ##################         
        x21        = Cropping3D(cropping = ((0,0),(20,20),(20,20)), input_shape=(None,None,None, self.num_channels),name = 'Sub_Detail')(Sub)
        
        # reduced original input by -40    : 13,35,35     

        for feature in self.conv_features[0:6]:    # reduce all dimensions in -12
            x21        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x21)
            x21        = LeakyReLU()(x21)                              
            x21        = BatchNormalization()(x21) 

        for feature in (self.conv_features[0:7]):    # reduce in -14
            x21        = Conv3D(filters = 40, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x21)
            x21        = LeakyReLU()(x21)                              
            x21        = BatchNormalization()(x21)  
        

        x2 = x21
        
#        ########################  T2 pathway #########################
#        #############   High res pathway   ##################         
#        x31        = Cropping3D(cropping = ((0,0),(20,20),(20,20)), input_shape=(None,None,None, self.num_channels),name = 'T2_Detail')(T2)
#        
#        # reduced original input by -40    : 13,35,35     
#
#        for feature in (self.conv_features[0:7]):    # reduce in -36
#            x31        = Conv3D(filters = feature, 
#                               kernel_size = (1,3,3), 
#                               #kernel_initializer=he_normal(seed=seed),
#                               kernel_initializer=Orthogonal(),
#                               kernel_regularizer=regularizers.l2(self.L2))(x31)
#            x31        = LeakyReLU()(x31)                              
#            x31        = BatchNormalization()(x31)     
#        
#        for feature in self.conv_features[0:6]:    # reduce all dimensions in -12
#            x31        = Conv3D(filters = feature, 
#                               kernel_size = (3,3,3), 
#                               #kernel_initializer=he_normal(seed=seed),
#                               kernel_initializer=Orthogonal(),
#                               kernel_regularizer=regularizers.l2(self.L2))(x31)
#            x31        = LeakyReLU()(x31)                              
#            x31        = BatchNormalization()(x31)    
#        
#        x3 = x31
        
	########################  Merge Modalities #########################

        TPM = Input((None,None,None,1), name='TPM')

        x = concatenate([x1,x2, TPM])   
          
        x        = Conv3D(filters = 150, 
                   kernel_size = (1,1,1), 
                   #kernel_initializer=he_normal(seed=seed),
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)

        x        = concatenate([x,TPM])  #  MIXING ONLY CHANNELS + CHANNELS. 
        
        x        = Conv3D(filters = 200, 
                   kernel_size = (1,1,1), 
                   #kernel_initializer=he_normal(seed=seed),
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
        
        
        x        = Conv3D(filters = 2, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = Activation('sigmoid')(x)
        
        model     = Model(inputs=[Context,T1post,Sub,TPM], outputs=x)

    	#model = multi_gpu_model(model, gpus=4)

    	if self.loss_function == 'Dice':
            model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        elif self.loss_function == 'Multinomial':
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        return model


class MultiPriors_v1():
    
    def __init__(self, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay, loss_function):
        
        self.output_classes = output_classes
        self.conv_features = [20,20,20,20,30,30,30,30,30,30,30,50,50,50,50] #[50,50,50,50,50,50,50,70,70,70,70,100,100]
        self.fc_features = [60,60,80,100]#[50,50,50,50]
        self.num_channels = num_channels
        self.L2 = L2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_decay = optimizer_decay
        self.loss_function = loss_function
    
    def createModel(self):
    
        Context = Input((None,None,None,1), name = 'V1_Context')
        T1post = Input((None,None,None, 1),name = 'V1_T1post_input')
        Sub = Input((None,None,None, 1),name = 'V1_Sub_input')
        Coords = Input((None,None,None,3), name = 'V1_Spatial_coordinates')
       
        #######################################################################
        ######################### Breast Mask Model ###########################
        #######################################################################
            
        x_mask = Context #AveragePooling3D(pool_size=(1, 3, 3), name='Context')(T1post)    
        # (13, 25, 25)        
        for iii in range(6):    
            x_mask        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               kernel_initializer=Orthogonal(),
                               name='V1_T1post_Context_{}'.format(iii),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)                              
            x_mask        = BatchNormalization(name='V1_BatchNorm_{}'.format(iii))(x_mask) 
        # (1, 13,13)
        for iii in range(5):    
            x_mask        = Conv3D(filters = 30, 
                               kernel_size = (1,3,3), 
                               kernel_initializer=Orthogonal(),
                               name='V1_T1post_Context_{}'.format(iii+6),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)                              
            x_mask        = BatchNormalization(name='V1_BatchNorm_{}'.format(iii+6))(x_mask)     
        # (1,3,3)
        x_mask   =  UpSampling3D(size=(1,3,3))(x_mask)
        # (1,9,9)
        ######################## FC Parts #############################
          
        x_mask = concatenate([x_mask, Coords])
        
        for iii in range(2):  
            x_mask        = Conv3D(filters = 60, 
                               kernel_size = (1,1,1), 
                               kernel_initializer=Orthogonal(),
                               name='V1_T1post_Context_{}'.format(iii+11),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)
            x_mask        = BatchNormalization(name='V1_BatchNorm_{}'.format(iii+11))(x_mask)
        
           
        x_mask        = Conv3D(filters = 100, 
                           kernel_size = (1,1,1), 
                           kernel_initializer=Orthogonal(),
                           name='V1_T1post_Context_13',
                           kernel_regularizer=regularizers.l2(self.L2))(x_mask)
        x_mask        = LeakyReLU()(x_mask)
        x_mask        = BatchNormalization(name='V1_BatchNorm_13')(x_mask)
        

        x_mask        = Conv3D(filters = 2, 
                           kernel_size = (1,1,1), 
                           name='V1_T1post_Context_14',
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(x_mask)
        x_mask        = Activation('sigmoid')(x_mask)
        
        #######################################################################
        #######################################################################
        #######################################################################        
        
        ########################  T1 post pathway #########################
        #############   High res pathway   ##################         
        x1  = Cropping3D(cropping = ((0,0),(20,20),(20,20)), input_shape=(None,None,None, self.num_channels),name = 'V1_T1post_Detail')(T1post)
        
        # (13,75,75)
        for feature in self.conv_features[0:6]:    # reduce all dimensions in -12
            x1        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = LeakyReLU()(x1)                              
            x1        = BatchNormalization()(x1)   
        # (1,63,63)
        for feature in (self.conv_features[0:7]):    # reduce in -36
            x1        = Conv3D(filters = 40, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = LeakyReLU()(x1)                              
            x1        = BatchNormalization()(x1)         
        # (13,75,75)            

           
        ########################  Sub pathway #########################
        #############   High res pathway   ##################         
        x2  = Cropping3D(cropping = ((0,0),(20,20),(20,20)), input_shape=(None,None,None, self.num_channels),name = 'V1_Sub_Detail')(Sub)
        
        # reduced original input by -40    : 13,35,35     

        for feature in self.conv_features[0:6]:    # reduce all dimensions in -12
            x2        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x2)
            x2        = LeakyReLU()(x2)                              
            x2        = BatchNormalization()(x2) 

        for feature in (self.conv_features[0:7]):    # reduce in -14
            x2        = Conv3D(filters = 40, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x2)
            x2        = LeakyReLU()(x2)                              
            x2        = BatchNormalization()(x2)  
        

        # 1, 23, 23
             
	########################  Merge Modalities #########################


        x = concatenate([x1,x2, Coords, x_mask])   
          
        x        = Conv3D(filters = 100, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)

                   
        x        = Conv3D(filters = 220, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
              
        x        = Conv3D(filters = 2, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = Activation('sigmoid')(x)
        
            
        model     = Model(inputs=[Context,T1post,Sub,Coords], outputs=x)

    	#model = multi_gpu_model(model, gpus=4)

    	if self.loss_function == 'Dice':
            model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        elif self.loss_function == 'Multinomial':
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        return model



class MultiPriors_v2():
    
    def __init__(self, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay, loss_function):
        
        self.output_classes = output_classes
        self.conv_features = [20,20,20,20,30,30,30,30,30,30,30,50,50,50,50] #[50,50,50,50,50,50,50,70,70,70,70,100,100]
        self.fc_features = [60,60,80,100]#[50,50,50,50]
        self.num_channels = num_channels
        self.L2 = L2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_decay = optimizer_decay
        self.loss_function = loss_function
    
    def createModel(self):
    
        Context = Input((None,None,None,1), name = 'V2_Context')
        T1pre = Input((None,None,None, 1),name = 'V2_T1pre_input')
        T1post = Input((None,None,None, 1),name = 'V2_T1post_input')
        Sub = Input((None,None,None, 1),name = 'V2_Sub_input')
        
        #Coords = Input((None,None,None,3), name = 'V2_Spatial_coordinates')
       
#######################################################################
        ######################### Breast Mask Model ###########################
        #######################################################################
            
        x_mask = Context #AveragePooling3D(pool_size=(1, 3, 3), name='Context')(T1post)    
        # (13, 25, 25)        
        #SEGMENTATION: (25,33,33)
        for iii in range(6):    
            x_mask        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               kernel_initializer=Orthogonal(),
                               name='V2_T1post_Context_{}'.format(iii),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)                              
            x_mask        = BatchNormalization(name='V2_BatchNorm_{}'.format(iii))(x_mask) 
        # (1, 13,13)
        #SEGMENTATION: (13,21,21)
        for iii in range(5):    
            x_mask        = Conv3D(filters = 30, 
                               kernel_size = (1,3,3), 
                               kernel_initializer=Orthogonal(),
                               name='V2_T1post_Context_{}'.format(iii+6),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)                              
            x_mask        = BatchNormalization(name='V2_BatchNorm_{}'.format(iii+6))(x_mask)     
        # (1,3,3)
        #SEGMENTATION: (13,11,11)
        x_mask   =  UpSampling3D(size=(1,3,3))(x_mask)
        # (1,9,9)
        #SEGMENTATION: (13,33,33)
        ######################## FC Parts #############################
          
        #x_mask = concatenate([x_mask, Coords])
        
        for iii in range(2):  
            x_mask        = Conv3D(filters =60, 
                               kernel_size = (1,1,1), 
                               kernel_initializer=Orthogonal(),
                               name='V2_T1post_Context_{}'.format(iii+11),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)
            x_mask        = BatchNormalization(name='V2_BatchNorm_{}'.format(iii+11))(x_mask)
        
           
        x_mask        = Conv3D(filters = 100, 
                           kernel_size = (1,1,1), 
                           kernel_initializer=Orthogonal(),
                           name='V2_T1post_Context_13',
                           kernel_regularizer=regularizers.l2(self.L2))(x_mask)
        x_mask        = LeakyReLU()(x_mask)
        x_mask        = BatchNormalization(name='V2_BatchNorm_13')(x_mask)
        

#        x_mask        = Conv3D(filters = 2, 
#                           kernel_size = (1,1,1), 
#                           name='T1post_Context_14',
#                           kernel_initializer=Orthogonal(),
#                           kernel_regularizer=regularizers.l2(self.L2))(x_mask)
#        x_mask        = Activation('sigmoid')(x_mask)
        
        #######################################################################
        #######################################################################
        #######################################################################        
        
        
        ########################  T1 pre pathway #########################
        #############   High res pathway   ##################         
        #x2 = Cropping3D(cropping = ((0,0),(20,20),(20,20)), input_shape=(None,None,None, self.num_channels),name = 'V2_Sub_Detail')(Sub)
        x0 = T1pre
        # reduced original input by -40    : 13,35,35     

        for feature in self.conv_features[0:6]:    # reduce all dimensions in -12
            x0        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x0)
            x0        = LeakyReLU()(x0)                              
            x0        = BatchNormalization()(x0) 

        for feature in (self.conv_features[0:7]):    # reduce in -14
            x0        = Conv3D(filters =40, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x0)
            x0        = LeakyReLU()(x0)                              
            x0        = BatchNormalization()(x0)  
        
        
        ########################  T1 post pathway #########################
        #############   High res pathway   ##################     
        
        #SEGMENTATION: (25,99,99)
        #x1      = Cropping3D(cropping = ((0,0),(20,20),(20,20)), input_shape=(None,None,None, self.num_channels),name = 'V2_T1post_Detail')(T1post)
        x1 = T1post
        #SEGMENTATION: (25,59,59)
        
        for feature in self.conv_features[0:6]:    # reduce all dimensions in -12
            x1        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = LeakyReLU()(x1)                              
            x1        = BatchNormalization()(x1)   

        # reduced original input by -40    : 13,35,35     
        #SEGMENTATION: (13,47,47)

        for feature in (self.conv_features[0:7]):    # reduce in -36
            x1        = Conv3D(filters = 40, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = LeakyReLU()(x1)                              
            x1        = BatchNormalization()(x1)         
            
        #SEGMENTATION: (13,33,33)

           
        ########################  T1 pre pathway #########################
        #############   High res pathway   ##################         
        #x2 = Cropping3D(cropping = ((0,0),(20,20),(20,20)), input_shape=(None,None,None, self.num_channels),name = 'V2_Sub_Detail')(Sub)
        x2 = Sub
        # reduced original input by -40    : 13,35,35     

        for feature in self.conv_features[0:6]:    # reduce all dimensions in -12
            x2        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x2)
            x2        = LeakyReLU()(x2)                              
            x2        = BatchNormalization()(x2) 

        for feature in (self.conv_features[0:7]):    # reduce in -14
            x2        = Conv3D(filters =40, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x2)
            x2        = LeakyReLU()(x2)                              
            x2        = BatchNormalization()(x2)  
        

        # 1, 23, 23
             
	########################  Merge Modalities #########################


        x = concatenate([x0, x1,x2])#, Coords])   
          
        x        = Conv3D(filters = 100, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
 
        x = concatenate([x, x_mask])   

                   
        x        = Conv3D(filters = 220, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
      
        x        = Conv3D(filters = 2, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = Activation('softmax')(x)  # LUKAS
        
        model     = Model(inputs=[Context,T1pre,T1post,Sub], outputs=x)

    	 #model = multi_gpu_model(model, gpus=4)
        if self.loss_function == 'Dice':
            model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])
        elif self.loss_function == 'ReLU_Dice':
            model.compile(loss=Generalised_dice_coef_multilabel2_ReLU, optimizer=RAdam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        elif self.loss_function == 'Multinomial':
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        return model



class MultiPriors_v2_Big():
    
    def __init__(self, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay, loss_function):
        
        self.output_classes = output_classes
        self.conv_features = [20,20,20,20,30,30,30,30,30,30,30,50,50,50,50] #[50,50,50,50,50,50,50,70,70,70,70,100,100]
        self.fc_features = [60,60,80,100]#[50,50,50,50]
        self.num_channels = num_channels
        self.L2 = L2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_decay = optimizer_decay
        self.loss_function = loss_function
    
    def createModel(self):
    
        Context = Input((None,None,None,1), name = 'V2_Context')
        T1post = Input((None,None,None, 1),name = 'V2_T1post_input')
        Sub = Input((None,None,None, 1),name = 'V2_Sub_input')
        Coords = Input((None,None,None,3), name = 'V2_Spatial_coordinates')
       
#######################################################################
        ######################### Breast Mask Model ###########################
        #######################################################################
        # Input: 13, 141, 141  (Context = [13, 47, 47])      
        x_mask = Context #AveragePooling3D(pool_size=(1, 3, 3), name='Context')(T1post)    
        # (13, 47, 47)        
        for iii in range(6):    
            x_mask        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)                              
            x_mask        = BatchNormalization()(x_mask) 
        # (1, 35,35)
        for iii in range(16):    
            x_mask        = Conv3D(filters = 30, 
                               kernel_size = (1,3,3), 
                               kernel_initializer=Orthogonal(),
                               
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)                              
            x_mask        = BatchNormalization()(x_mask)     
        # (1,3,3)
        x_mask   =  UpSampling3D(size=(1,3,3))(x_mask)
        # (1,9,9)
        ######################## FC Parts #############################
          
        x_mask = concatenate([x_mask, Coords])
        
        for iii in range(2):  
            x_mask        = Conv3D(filters = 50, 
                               kernel_size = (1,1,1), 
                               kernel_initializer=Orthogonal(),
                               #name='V2_T1post_Context_{}'.format(iii+22),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)
            x_mask        = BatchNormalization()(x_mask)
        
           
        x_mask        = Conv3D(filters = 100, 
                           kernel_size = (1,1,1), 
                           kernel_initializer=Orthogonal(),
                           #name='V2_T1post_Context_13',
                           kernel_regularizer=regularizers.l2(self.L2))(x_mask)
        x_mask        = LeakyReLU()(x_mask)
        x_mask        = BatchNormalization()(x_mask)
        

#        x_mask        = Conv3D(filters = 2, 
#                           kernel_size = (1,1,1), 
#                           name='T1post_Context_14',
#                           kernel_initializer=Orthogonal(),
#                           kernel_regularizer=regularizers.l2(self.L2))(x_mask)
#        x_mask        = Activation('sigmoid')(x_mask)
        
        #######################################################################
        #######################################################################
        #######################################################################        
        
        ########################  T1 post pathway #########################
        #############   High res pathway   ##################         
        # Input: 13, 141, 141
        #x1      = Cropping3D(cropping = ((0,0),(35,35),(35,35)), input_shape=(None,None,None, self.num_channels),name = 'V2_T1post_Detail')(T1post)
        #x1 = T1post
        x1 = concatenate([T1post, Sub])
        # (13, 71, 71)
        for feature in range(6):    # reduce all dimensions in -12
            x1        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = LeakyReLU()(x1)                              
            x1        = BatchNormalization()(x1)   
        # (1, 59, 59)

        for feature in range(25):    # reduce in -50
            x1        = Conv3D(filters = 50, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = LeakyReLU()(x1)                              
            x1        = BatchNormalization()(x1)         
            
        # (1, 9, 9)
           
#        ########################  T1 pre pathway #########################
#        #############   High res pathway   ##################         
#        #x2 = Cropping3D(cropping = ((0,0),(35,35),(35,35)), input_shape=(None,None,None, self.num_channels),name = 'V2_Sub_Detail')(Sub)
#        x2 = Sub
#        # (13, 71, 71)
#        for feature in range(6):    # reduce all dimensions in -12
#            x2        = Conv3D(filters = 30, 
#                               kernel_size = (3,3,3), 
#                               #kernel_initializer=he_normal(seed=seed),
#                               kernel_initializer=Orthogonal(),
#                               kernel_regularizer=regularizers.l2(self.L2))(x2)
#            x2        = LeakyReLU()(x2)                              
#            x2        = BatchNormalization()(x2)   
#        # (1, 59, 59)
#
#        for feature in range(25):    # reduce in -80
#            x2        = Conv3D(filters = 40, 
#                               kernel_size = (1,3,3), 
#                               #kernel_initializer=he_normal(seed=seed),
#                               kernel_initializer=Orthogonal(),
#                               kernel_regularizer=regularizers.l2(self.L2))(x2)
#            x2        = LeakyReLU()(x2)                              
#            x2        = BatchNormalization()(x2)         
#            
#        # (1, 9, 9)
#             
	########################  Merge Modalities #########################


#        x = concatenate([x1,x2, Coords])   
        x = concatenate([x1, Coords, x_mask])
          
        x        = Conv3D(filters = 100, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
 
        x = concatenate([x, Coords, x_mask])   

                   
        x        = Conv3D(filters = 200, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
      
        x        = Conv3D(filters = 2, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = Activation('softmax')(x)  # LUKAS
        
        model     = Model(inputs=[Context,T1post,Sub,Coords], outputs=x)
        model = multi_gpu_model(model, gpus=4)
        if self.loss_function == 'Dice':
            model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])
        elif self.loss_function == 'ReLU_Dice':
            model.compile(loss=Generalised_dice_coef_multilabel2_ReLU, optimizer=RAdam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        elif self.loss_function == 'Multinomial':
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        return model



class MultiPriors_v2_Big_U():
    
    def __init__(self, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay, loss_function):
        
        self.output_classes = output_classes
        self.conv_features = [20,20,20,20,30,30,30,30,30,30,30,50,50,50,50] #[50,50,50,50,50,50,50,70,70,70,70,100,100]
        self.fc_features = [60,60,80,100]#[50,50,50,50]
        self.num_channels = num_channels
        self.L2 = L2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_decay = optimizer_decay
        self.loss_function = loss_function
    
    def createModel(self):
    
        Context = Input((None,None,None,1), name = 'V2_Context')
        T1post = Input((None,None,None, 1),name = 'V2_T1post_input')
        Sub = Input((None,None,None, 1),name = 'V2_Sub_input')
        Coords = Input((None,None,None,3), name = 'V2_Spatial_coordinates')
       
#######################################################################
        ######################### Breast Mask Model ###########################
        #######################################################################
        # Input: 13, 141, 141  (Context = [13, 45, 45])      
        x_mask = Context #AveragePooling3D(pool_size=(1, 3, 3), name='Context')(T1post)    
        # (13, 45, 45)        
        for iii in range(6):    
            x_mask        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)                              
            x_mask        = BatchNormalization()(x_mask) 
        for iii in range(16):    
            x_mask        = Conv3D(filters = 30, 
                               kernel_size = (1,3,3), 
                               kernel_initializer=Orthogonal(),
                               
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)                              
            x_mask        = BatchNormalization()(x_mask)     
        # (1,1,1)
        x_mask   =  UpSampling3D(size=(1,3,3))(x_mask)    
        
        x_mask        = Conv3D(filters = 50, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x_mask)
        x_mask        = LeakyReLU()(x_mask)
        x_mask        = BatchNormalization()(x_mask)        
        # (1,9,9)
        ########################  T1 post pathway #########################
        #############   High res pathway   ##################         
        x1 = concatenate([T1post, Sub])
        # (13, 65, 65)
        for feature in range(6):    # reduce all dimensions in -12
            x1        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = LeakyReLU()(x1)                              
            x1        = BatchNormalization()(x1)   

        for feature in range(25):    # reduce in -50
            x1        = Conv3D(filters = 50, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = LeakyReLU()(x1)                              
            x1        = BatchNormalization()(x1)         
        #####################################################           
        # (1,3,3)
#        x = concatenate([x1,x2, Coords])   
        x = concatenate([x1, Coords, x_mask])
          
        x        = Conv3D(filters = 50, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
        x   =  UpSampling3D(size=(1,2,2))(x)
        
        #(1,6,6)

        x        = Conv3D(filters = 60, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
        x   =  UpSampling3D(size=(1,2,2))(x)
        
        #(1,12, 12)
        x        = Conv3D(filters = 60, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
        x   =  UpSampling3D(size=(1,2,2))(x)        
        #(1, 24, 24)

        x        = Conv3D(filters = 60, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
        x   =  UpSampling3D(size=(1,2,2))(x)             
        
        # (1,48,48)
        
        x        = Conv3D(filters = 80, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
        
        x        = Conv3D(filters = 2, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = Activation('softmax')(x)  # LUKAS
        
        model     = Model(inputs=[Context,T1post,Sub,Coords], outputs=x)
        model = multi_gpu_model(model, gpus=4)
        if self.loss_function == 'Dice':
            model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])
        elif self.loss_function == 'ReLU_Dice':
            model.compile(loss=Generalised_dice_coef_multilabel2_ReLU, optimizer=RAdam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        elif self.loss_function == 'Multinomial':
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        return model



class MultiPriors_v2_Big_BreastMask():
    
    def __init__(self, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay, loss_function):
        
        self.output_classes = output_classes
        self.conv_features = [20,20,20,20,30,30,30,30,30,30,30,50,50,50,50] #[50,50,50,50,50,50,50,70,70,70,70,100,100]
        self.fc_features = [60,60,80,100]#[50,50,50,50]
        self.num_channels = num_channels
        self.L2 = L2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_decay = optimizer_decay
        self.loss_function = loss_function
    
    def createModel(self):
    
        Context = Input((None,None,None,1), name = 'V2_Context')
        T1post = Input((None,None,None, 1),name = 'V2_T1post_input')
        Sub = Input((None,None,None, 1),name = 'V2_Sub_input')
        Coords = Input((None,None,None,3), name = 'V2_Spatial_coordinates')
       
#######################################################################
        
        ######################### Breast Mask Model ###########################
        #######################################################################
        # Before: 75 --> 9  (-66)
        #(13,141,141) --> 9
        x_mask = Cropping3D(cropping = ((0,0),(11,11),(11,11)), input_shape=(None,None,None, self.num_channels), name='Context')(Context)    
        # (13, 141, 141)        
        #SEGMENTATION: (25,33,33)
        for iii in range(6):    
            x_mask        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               kernel_initializer=Orthogonal(),
                               name='V2_T1post_Context_{}'.format(iii),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)                              
            x_mask        = BatchNormalization(name='V2_BatchNorm_{}'.format(iii))(x_mask) 
        # (1, 13,13)
        #SEGMENTATION: (13,21,21)
        for iii in range(5):    
            x_mask        = Conv3D(filters = 30, 
                               kernel_size = (1,3,3), 
                               kernel_initializer=Orthogonal(),
                               name='V2_T1post_Context_{}'.format(iii+6),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)                              
            x_mask        = BatchNormalization(name='V2_BatchNorm_{}'.format(iii+6))(x_mask)     
        # (1,3,3)
        #SEGMENTATION: (13,11,11)
        x_mask   =  UpSampling3D(size=(1,3,3))(x_mask)
        # (1,9,9)
        #SEGMENTATION: (13,33,33)
        ######################## FC Parts #############################
          
        x_mask = concatenate([x_mask, Coords])
        
        for iii in range(2):  
            x_mask        = Conv3D(filters = 60, 
                               kernel_size = (1,1,1), 
                               kernel_initializer=Orthogonal(),
                               name='V2_T1post_Context_{}'.format(iii+11),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)
            x_mask        = BatchNormalization(name='V2_BatchNorm_{}'.format(iii+11))(x_mask)
        
           
        x_mask        = Conv3D(filters = 100, 
                           kernel_size = (1,1,1), 
                           kernel_initializer=Orthogonal(),
                           name='V2_T1post_Context_13',
                           kernel_regularizer=regularizers.l2(self.L2))(x_mask)
        x_mask        = LeakyReLU()(x_mask)
        x_mask        = BatchNormalization(name='V2_BatchNorm_13')(x_mask)
        

        x_mask        = Conv3D(filters = 2, 
                           kernel_size = (1,1,1), 
                           name='T1post_Context_14',
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(x_mask)
        x_mask        = Activation('sigmoid')(x_mask)
        
        #######################################################################
        #######################################################################
        #######################################################################        
        
        ########################  T1 post pathway #########################
        #############   High res pathway   ##################         
        # Input: 13, 141, 141
        #x1      = Cropping3D(cropping = ((0,0),(35,35),(35,35)), input_shape=(None,None,None, self.num_channels),name = 'V2_T1post_Detail')(T1post)
        #x1 = T1post
        
        x1 = concatenate([T1post, Sub])
        
        # (13, 71, 71)
        for feature in range(6):    # reduce all dimensions in -12
            x1        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = LeakyReLU()(x1)                              
            x1        = BatchNormalization()(x1)   
        # (1, 59, 59)

        for feature in range(25):    # reduce in -50
            x1        = Conv3D(filters = 40, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = LeakyReLU()(x1)                              
            x1        = BatchNormalization()(x1)         
            
        # (1, 9, 9)
           
#        ########################  T1 pre pathway #########################
#        #############   High res pathway   ##################         
#        #x2 = Cropping3D(cropping = ((0,0),(35,35),(35,35)), input_shape=(None,None,None, self.num_channels),name = 'V2_Sub_Detail')(Sub)
#        x2 = Sub
#        # (13, 71, 71)
#        for feature in range(6):    # reduce all dimensions in -12
#            x2        = Conv3D(filters = 30, 
#                               kernel_size = (3,3,3), 
#                               #kernel_initializer=he_normal(seed=seed),
#                               kernel_initializer=Orthogonal(),
#                               kernel_regularizer=regularizers.l2(self.L2))(x2)
#            x2        = LeakyReLU()(x2)                              
#            x2        = BatchNormalization()(x2)   
#        # (1, 59, 59)
#
#        for feature in range(25):    # reduce in -80
#            x2        = Conv3D(filters = 40, 
#                               kernel_size = (1,3,3), 
#                               #kernel_initializer=he_normal(seed=seed),
#                               kernel_initializer=Orthogonal(),
#                               kernel_regularizer=regularizers.l2(self.L2))(x2)
#            x2        = LeakyReLU()(x2)                              
#            x2        = BatchNormalization()(x2)         
#            
#        # (1, 9, 9)
             
	########################  Merge Modalities #########################


        #x = concatenate([x1,x2, Coords, x_mask])   
         
        x = concatenate([x1, Coords, x_mask])    
        
        x        = Conv3D(filters = 100, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
 
        x = concatenate([x, Coords, x_mask])   

                   
        x        = Conv3D(filters = 200, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
      
        
        x        = Conv3D(filters = 2, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
        
        x        = Activation('softmax')(x)  # LUKAS
        
        model     = Model(inputs=[Context,T1post,Sub,Coords], outputs=x)
        #model = multi_gpu_model(model, gpus=4)
#        if self.loss_function == 'Dice':
#            model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])
#        elif self.loss_function == 'ReLU_Dice':
#            model.compile(loss=Generalised_dice_coef_multilabel2_ReLU, optimizer=RAdam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
#        elif self.loss_function == 'Multinomial':
#            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        return model


class MultiPriors_v2_ContextOutput():
    
    def __init__(self, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay, loss_function):
        
        self.output_classes = output_classes
        self.conv_features = [20,20,20,20,30,30,30,30,30,30,30,50,50,50,50] #[50,50,50,50,50,50,50,70,70,70,70,100,100]
        self.fc_features = [60,60,80,100]#[50,50,50,50]
        self.num_channels = num_channels
        self.L2 = L2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_decay = optimizer_decay
        self.loss_function = loss_function
    
    def createModel(self):
    
        Context = Input((None,None,None,1), name = 'Context')
        T1post = Input((None,None,None, 1),name = 'T1post_input')
        Sub = Input((None,None,None, 1),name = 'Sub_input')
        Coords = Input((None,None,None,3), name = 'Spatial_coordinates')
       
#######################################################################
        ######################### Breast Mask Model ###########################
        #######################################################################
            
        x_mask = Context #AveragePooling3D(pool_size=(1, 3, 3), name='Context')(T1post)    
        # (13, 25, 25)        
        for iii in range(6):    
            x_mask        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               kernel_initializer=Orthogonal(),
                               name='T1post_Context_{}'.format(iii),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)                              
            x_mask        = BatchNormalization(name='BatchNorm_{}'.format(iii))(x_mask) 
        # (1, 13,13)
        for iii in range(5):    
            x_mask        = Conv3D(filters = 30, 
                               kernel_size = (1,3,3), 
                               kernel_initializer=Orthogonal(),
                               name='T1post_Context_{}'.format(iii+6),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)                              
            x_mask        = BatchNormalization(name='BatchNorm_{}'.format(iii+6))(x_mask)     
        # (1,3,3)
        x_mask   =  UpSampling3D(size=(1,3,3))(x_mask)
        # (1,9,9)
        ######################## FC Parts #############################
          
        x_mask = concatenate([x_mask, Coords])
        
        for iii in range(2):  
            x_mask        = Conv3D(filters = 60, 
                               kernel_size = (1,1,1), 
                               kernel_initializer=Orthogonal(),
                               name='T1post_Context_{}'.format(iii+11),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)
            x_mask        = BatchNormalization(name='BatchNorm_{}'.format(iii+11))(x_mask)
        
           
        x_mask        = Conv3D(filters = 100, 
                           kernel_size = (1,1,1), 
                           kernel_initializer=Orthogonal(),
                           name='T1post_Context_13',
                           kernel_regularizer=regularizers.l2(self.L2))(x_mask)
        x_mask        = LeakyReLU()(x_mask)
        x_mask        = BatchNormalization(name='BatchNorm_13')(x_mask)
        

#        x_mask        = Conv3D(filters = 2, 
#                           kernel_size = (1,1,1), 
#                           name='T1post_Context_14',
#                           kernel_initializer=Orthogonal(),
#                           kernel_regularizer=regularizers.l2(self.L2))(x_mask)
#        x_mask        = Activation('sigmoid')(x_mask)
        
        #######################################################################
        #######################################################################
        #######################################################################        
        
        ########################  T1 post pathway #########################
        #############   High res pathway   ##################         
        x1      = Cropping3D(cropping = ((0,0),(20,20),(20,20)), input_shape=(None,None,None, self.num_channels),name = 'T1post_Detail')(T1post)
        
        for feature in self.conv_features[0:6]:    # reduce all dimensions in -12
            x1        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = LeakyReLU()(x1)                              
            x1        = BatchNormalization()(x1)   

        # reduced original input by -40    : 13,35,35     
        for feature in (self.conv_features[0:7]):    # reduce in -36
            x1        = Conv3D(filters = 40, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = LeakyReLU()(x1)                              
            x1        = BatchNormalization()(x1)         
            

           
        ########################  T1 pre pathway #########################
        #############   High res pathway   ##################         
        x2 = Cropping3D(cropping = ((0,0),(20,20),(20,20)), input_shape=(None,None,None, self.num_channels),name = 'Sub_Detail')(Sub)
        
        # reduced original input by -40    : 13,35,35     

        for feature in self.conv_features[0:6]:    # reduce all dimensions in -12
            x2        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x2)
            x2        = LeakyReLU()(x2)                              
            x2        = BatchNormalization()(x2) 

        for feature in (self.conv_features[0:7]):    # reduce in -14
            x2        = Conv3D(filters = 40, 
                               kernel_size = (1,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x2)
            x2        = LeakyReLU()(x2)                              
            x2        = BatchNormalization()(x2)  
        

        # 1, 23, 23
             
	########################  Merge Modalities #########################


        x = concatenate([x1,x2, Coords])   
          
        x        = Conv3D(filters = 100, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
 
        x = concatenate([x, x_mask])   

                   
        x        = Conv3D(filters = 220, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
      
        x        = Conv3D(filters = 2, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = Activation('sigmoid')(x)
        
        model     = Model(inputs=[Context,T1post,Sub,Coords], outputs=x_mask)

    	#model = multi_gpu_model(model, gpus=4)

    	if self.loss_function == 'Dice':
            model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        elif self.loss_function == 'Multinomial':
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        return model



class MultiPriors_v3_TEST():
    
    def __init__(self, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay, loss_function):
        
        self.output_classes = output_classes
        self.conv_features = [20,20,20,20,30,30,30,30,30,30,30,50,50,50,50] #[50,50,50,50,50,50,50,70,70,70,70,100,100]
        self.fc_features = [60,60,80,100]#[50,50,50,50]
        self.num_channels = num_channels
        self.L2 = L2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_decay = optimizer_decay
        self.loss_function = loss_function
    
    def createModel(self):
    
        Context = Input((None,None,None,1), name = 'Context')
        T1post = Input((None,None,None, 1),name = 'T1post_input')
        Sub = Input((None,None,None, 1),name = 'Sub_input')
        Coords = Input((None,None,None,3), name = 'Spatial_coordinates')
       
        #######################################################################
        ######################### Breast Mask Model ###########################
        #######################################################################
            
        x_mask = AveragePooling3D(pool_size=(1, 3, 3), name='Context')(T1post)    
        # (13, 25, 25)        
        for iii in range(6):    
            x_mask        = Conv3D(filters = 30, 
                               kernel_size = (3,3,3), 
                               kernel_initializer=Orthogonal(),
                               name='T1post_Context_{}'.format(iii),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)                              
            x_mask        = BatchNormalization(name='BatchNorm_{}'.format(iii))(x_mask) 
        # (1, 13,13)
        for iii in range(5):    
            x_mask        = Conv3D(filters = 30, 
                               kernel_size = (1,3,3), 
                               kernel_initializer=Orthogonal(),
                               name='T1post_Context_{}'.format(iii+6),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)                              
            x_mask        = BatchNormalization(name='BatchNorm_{}'.format(iii+6))(x_mask)     
        # (1,3,3)
        x_mask   =  UpSampling3D(size=(1,3,3))(x_mask)
        # (1,9,9)
        ######################## FC Parts #############################
          
        x_mask = concatenate([x_mask, Coords])
        
        for iii in range(2):  
            x_mask        = Conv3D(filters = 60, 
                               kernel_size = (1,1,1), 
                               kernel_initializer=Orthogonal(),
                               name='T1post_Context_{}'.format(iii+11),
                               kernel_regularizer=regularizers.l2(self.L2))(x_mask)
            x_mask        = LeakyReLU()(x_mask)
            x_mask        = BatchNormalization(name='BatchNorm_{}'.format(iii+11))(x_mask)
        
           
        x_mask        = Conv3D(filters = 100, 
                           kernel_size = (1,1,1), 
                           kernel_initializer=Orthogonal(),
                           name='T1post_Context_13',
                           kernel_regularizer=regularizers.l2(self.L2))(x_mask)
        x_mask        = LeakyReLU()(x_mask)
        x_mask        = BatchNormalization(name='BatchNorm_13')(x_mask)
        

        x_mask        = Conv3D(filters = 2, 
                           kernel_size = (1,1,1), 
                           name='T1post_Context_14',
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(x_mask)
        x_mask        = Activation('sigmoid')(x_mask)
#        #######################################################################
#        #######################################################################
#        #######################################################################        
#        
#        ########################  T1 post pathway #########################
#        #############   High res pathway   ##################         
#        x1  = Cropping3D(cropping = ((0,0),(20,20),(20,20)), input_shape=(None,None,None, self.num_channels),name = 'T1post_Detail')(T1post)
#        
#        # (13,75,75)
#        for feature in self.conv_features[0:6]:    # reduce all dimensions in -12
#            x1        = Conv3D(filters = 4, 
#                               kernel_size = (3,3,3), 
#                               #kernel_initializer=he_normal(seed=seed),
#                               kernel_initializer=Orthogonal(),
#                               kernel_regularizer=regularizers.l2(self.L2))(x1)
#            x1        = LeakyReLU()(x1)                              
#            x1        = BatchNormalization()(x1)   
#        # (1,63,63)
#        for feature in (self.conv_features[0:7]):    # reduce in -36
#            x1        = Conv3D(filters = 4, 
#                               kernel_size = (1,3,3), 
#                               #kernel_initializer=he_normal(seed=seed),
#                               kernel_initializer=Orthogonal(),
#                               kernel_regularizer=regularizers.l2(self.L2))(x1)
#            x1        = LeakyReLU()(x1)                              
#            x1        = BatchNormalization()(x1)         
#        # (13,75,75)            
#
#           
#        ########################  Sub pathway #########################
#        #############   High res pathway   ##################         
#        x2  = Cropping3D(cropping = ((0,0),(20,20),(20,20)), input_shape=(None,None,None, self.num_channels),name = 'Sub_Detail')(Sub)
#        
#        # reduced original input by -40    : 13,35,35     
#
#        for feature in self.conv_features[0:6]:    # reduce all dimensions in -12
#            x2        = Conv3D(filters = 4, 
#                               kernel_size = (3,3,3), 
#                               #kernel_initializer=he_normal(seed=seed),
#                               kernel_initializer=Orthogonal(),
#                               kernel_regularizer=regularizers.l2(self.L2))(x2)
#            x2        = LeakyReLU()(x2)                              
#            x2        = BatchNormalization()(x2) 
#
#        for feature in (self.conv_features[0:7]):    # reduce in -14
#            x2        = Conv3D(filters = 4, 
#                               kernel_size = (1,3,3), 
#                               #kernel_initializer=he_normal(seed=seed),
#                               kernel_initializer=Orthogonal(),
#                               kernel_regularizer=regularizers.l2(self.L2))(x2)
#            x2        = LeakyReLU()(x2)                              
#            x2        = BatchNormalization()(x2)  
#        
#
#        # 1, 23, 23
#             
#	########################  Merge Modalities #########################
#
#
#        x = concatenate([x1,x2, Coords, x_mask])   
#          
#        x        = Conv3D(filters = 4, 
#                   kernel_size = (1,1,1), 
#                   kernel_initializer=Orthogonal(),
#                   kernel_regularizer=regularizers.l2(self.L2))(x)
#        x        = LeakyReLU()(x)
#        x        = BatchNormalization()(x)
#     
        
        model     = Model(inputs=[Context,T1post,Sub,Coords], outputs=x_mask)

    	#model = multi_gpu_model(model, gpus=4)

    	if self.loss_function == 'Dice':
            model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        elif self.loss_function == 'Multinomial':
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['acc', dice_coef_multilabel0, dice_coef_multilabel1])
        return model

# Debug
    
#mp = MultiPriors_v0(2, 3, 0.001, [0], 0.01, 0, 'Dice' )
#model = mp.createModel()            
#model.summary()  
#from keras.utils import plot_model
#plot_model(model, to_file='/home/deeperthought/Projects/MultiPriors_MSKCC/' +'/MultiPriors_v0.png', show_shapes=True)    
#X = np.random.randn(1,13,75,75,1)
#y = np.random.binomial(n=1, p=0.5,size=81*2).reshape(1,1,9,9,2)
#TPM = np.random.randn(1,9,9,1,1)
#yhat = model.predict([X,X,X,TPM])
#print(yhat.shape)


#mp = MultiPriors_v2(2, 3, 0.001, [0], 0.01, 0, 'Dice' )
#model = mp.createModel()     
#model.summary()  
#from keras.utils import plot_model
#plot_model(model, to_file='/home/deeperthought/Projects/MultiPriors_MSKCC/' +'/MultiPriors_v1.png', show_shapes=True)    
#X = np.random.randn(1,13,75,75,1)
#Context = np.random.randn(1,13,25,25,1)
#y = np.random.binomial(n=1, p=0.5,size=81*2).reshape(1,1,9,9,2)
#coords = np.random.randn(1,1,9,9,3)
#yhat = model.predict([Context, X, X, coords])
#print(yhat.shape)
#model.fit([Context,X, X, coords], y, epochs=10)


#mp = MultiPriors_v3_TEST(2, 3, 0.001, [0], 0.01, 0, 'Dice' )
#model = mp.createModel()     
#model.summary()  
#from keras.utils import plot_model
#plot_model(model, to_file='/home/deeperthought/Projects/MultiPriors_MSKCC/' +'/MultiPriors_v1.png', show_shapes=True)    
#X = np.random.randn(1,13,75,75,1)
#Context = np.random.randn(1,13,25,25,1)
#y = np.random.binomial(n=1, p=0.5,size=81*2).reshape(1,1,9,9,2)
#coords = np.random.randn(1,1,9,9,3)
#yhat = model.predict([Context, X, X, coords])
#print(yhat.shape)
#model.fit([Context,X, X, coords], y, epochs=10)
        
    
#mp = MultiPriors_v2_Big(2, 3, 0.001, [0], 0.01, 0, 'Dice' )
#model = mp.createModel()     
#model.summary()  
#
#from keras.utils import plot_model
#plot_model(model, to_file='/home/deeperthought/Projects/MultiPriors_MSKCC/' +'/MultiPriors_v2_BIG.png', show_shapes=True)    
#model = multi_gpu_model(model, gpus=4)
#model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(0.0001), metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])
#
#N = 1000
#X = np.random.randn(N,13,141,141,1)
#Context = np.random.randn(N,13,47,47,1)
#y = np.random.binomial(n=1, p=0.5,size=81*2*N).reshape(N,1,9,9,2)
#coords = np.random.randn(N,1,9,9,3)
#yhat = model.predict([Context, X, X, coords])
#print(yhat.shape)
#model.fit([Context,X, X, coords], y, epochs=2, batch_size=16)
