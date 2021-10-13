#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:31:09 2021

@author: deeperthought
"""


import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/deeperthought/Projects/MultiPriors_MSKCC/scripts/')
from lib import *
from scipy.ndimage.measurements import label

import tensorflow.keras as keras

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0,1,2,3"
tf.keras.backend.set_session(tf.Session(config=config))

n_base_filters = 32
learning_rate = 1e-6
    

#%%

    
from scipy.ndimage import gaussian_filter1d 

def mix_lists(l1, l2):
    """ Mix two lists evenly, generating the elements of the mixed list. """
    m, n = len(l1), len(l2)
    for i in range(m + n):
        q, r = divmod(i * n, m + n)
        yield l1[i - q] if r < m else l2[q]
        
        
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    

def train_model(model_name, model, INPUT_SHAPE, INPUT_SHAPE_VAL, MODEL_REDUCTION, BATCH_SIZE, FULL_SLICE_REDUCTION, history_callback, history_dict, X_train, Y_train, X_val_full_slice, Y_val_full_slice, X_val_patches, Y_val_patches, SHUFFLE=False):
    print('------ {} -------'.format(model_name))    
    DATA_SHAPE = X_train.shape[-2]
    OUTPUT_SHAPE = INPUT_SHAPE - MODEL_REDUCTION    
    
    model.fit(X_train[:,:,DATA_SHAPE/2-INPUT_SHAPE/2:DATA_SHAPE/2+INPUT_SHAPE/2+1,
                          DATA_SHAPE/2-INPUT_SHAPE/2:DATA_SHAPE/2+INPUT_SHAPE/2+1,:], 
              Y_train[:,:,DATA_SHAPE/2-OUTPUT_SHAPE/2:DATA_SHAPE/2+OUTPUT_SHAPE/2+1,
                          DATA_SHAPE/2-OUTPUT_SHAPE/2:DATA_SHAPE/2+OUTPUT_SHAPE/2+1,:], 
              epochs = 1, batch_size=BATCH_SIZE, callbacks = [history_callback], shuffle=SHUFFLE)
    history_dict['loss'].extend(history_callback.losses)
    history_dict['metrics'].append(history_callback.metrics)

    INPUT_SHAPE = X_val_full_slice.shape[-2]-FULL_SLICE_REDUCTION
    OUTPUT_SHAPE = INPUT_SHAPE - MODEL_REDUCTION
    val = model.evaluate(X_val_full_slice[:,:,507/2-INPUT_SHAPE/2:507/2+INPUT_SHAPE/2+1,507/2-INPUT_SHAPE/2:507/2+INPUT_SHAPE/2+1,:], 
                         Y_val_full_slice[:,:,507/2-OUTPUT_SHAPE/2:507/2+OUTPUT_SHAPE/2+1,507/2-OUTPUT_SHAPE/2:507/2+OUTPUT_SHAPE/2+1,:], 
                         batch_size=1, verbose=True)
    print(val)
    history_dict['val_full_slice_loss'].append(val[0])
    history_dict['val_full_slice_metrics'].append(val[2:])   

    DATA_SHAPE = X_val_patches.shape[-2]
    OUTPUT_SHAPE = INPUT_SHAPE_VAL - MODEL_REDUCTION
    val = model.evaluate(X_val_patches[:,:,DATA_SHAPE/2-INPUT_SHAPE_VAL/2:DATA_SHAPE/2+INPUT_SHAPE_VAL/2+1,
                                           DATA_SHAPE/2-INPUT_SHAPE_VAL/2:DATA_SHAPE/2+INPUT_SHAPE_VAL/2+1,:],  
                         Y_val_patches[:,:,DATA_SHAPE/2-OUTPUT_SHAPE/2:DATA_SHAPE/2+OUTPUT_SHAPE/2+1,
                                           DATA_SHAPE/2-OUTPUT_SHAPE/2:DATA_SHAPE/2+OUTPUT_SHAPE/2+1,:], 
                         batch_size=BATCH_SIZE, verbose=True)
    print(val)
    history_dict['val_patches_loss'].append(val[0])
    history_dict['val_patches_metrics'].append(val[2:])  
    
    LAST = history_dict['val_full_slice_metrics'][-1][1]
    if LAST > 0.606:
        MAX = np.max([d[1] for d in history_dict['val_full_slice_metrics']])
        if LAST == MAX:
            wd = '/home/deeperthought/Projects/MultiPriors_MSKCC/models/Patch_training/best_models/'          
            model.save_weights(wd + model_name + '_weights.h5')        
            model.save(wd + model_name + '_model.h5')        

    
    
def val_model(model_name, model, INPUT_SHAPE, MODEL_REDUCTION, VAL_BATCH, FULL_SLICE_REDUCTION, history_callback, history_dict, X_val_full_slice, Y_val_full_slice, X_val_patches, Y_val_patches):
    print('------ {} -------'.format(model_name))    
    DATA_SHAPE = X_val_full_slice.shape[-2]-FULL_SLICE_REDUCTION
    OUTPUT_SHAPE = DATA_SHAPE - MODEL_REDUCTION
    val = model.evaluate(X_val_full_slice[:,:,507/2-DATA_SHAPE/2:507/2+DATA_SHAPE/2+1,507/2-DATA_SHAPE/2:507/2+DATA_SHAPE/2+1,:], 
                                  Y_val_full_slice[:,:,507/2-OUTPUT_SHAPE/2:507/2+OUTPUT_SHAPE/2+1,507/2-OUTPUT_SHAPE/2:507/2+OUTPUT_SHAPE/2+1,:], 
                                  batch_size=1, verbose=True)
    print(val)
    history_dict['val_full_slice_loss'].append(val[0])
    history_dict['val_full_slice_metrics'].append(val[2:])   

    DATA_SHAPE = X_val_patches.shape[-2]
    OUTPUT_SHAPE = INPUT_SHAPE - MODEL_REDUCTION
    val = model.evaluate(X_val_patches[:,:,DATA_SHAPE/2-INPUT_SHAPE/2:DATA_SHAPE/2+INPUT_SHAPE/2+1,
                                           DATA_SHAPE/2-INPUT_SHAPE/2:DATA_SHAPE/2+INPUT_SHAPE/2+1,:],  
                         Y_val_patches[:,:,DATA_SHAPE/2-OUTPUT_SHAPE/2:DATA_SHAPE/2+OUTPUT_SHAPE/2+1,
                                           DATA_SHAPE/2-OUTPUT_SHAPE/2:DATA_SHAPE/2+OUTPUT_SHAPE/2+1,:], 
                         batch_size=VAL_BATCH, verbose=True)
    print(val)
    history_dict['val_patches_loss'].append(val[0])
    history_dict['val_patches_metrics'].append(val[2:])  



from Unet_3D_Class import UNet_v4, UNet_v4_B
from Unet_3D_Class import UNet_v5, UNet_v5_2lyr_deeper
from Unet_3D_Class import UNet_v4_deeper


from tensorflow.keras.models import load_model   
from MultiPriors_Models_Collection import Generalised_dice_coef_multilabel2, dice_coef_multilabel_bin0,dice_coef_multilabel_bin1
my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
                     'dice_coef_multilabel_bin0':dice_coef_multilabel_bin0,
                     'dice_coef_multilabel_bin1':dice_coef_multilabel_bin1}


def plot_training(training_history, SIGMA, VAL_SIGMA, NAME, SAVE, OUTPUT):
    plt.figure(figsize=(15,10))
    plt.suptitle(NAME)
    
    plt.subplot(341)
    plt.ylabel('Train loss')
    plt.plot(gaussian_filter1d(training_history['loss'], SIGMA))
    plt.grid()
  
    trainmetric0 = [ ]
    trainmetric1 = [ ]

    plt.subplot(342)
    trainmetrics = np.array(training_history['metrics'])
    for x in trainmetrics:
        for y in x:
            trainmetric0.append(y[0])
            trainmetric1.append(y[1])
    

    #trainmetrics = trainmetrics.reshape((-1,2))
    
    plt.ylabel('Train Dice')
    plt.plot(gaussian_filter1d(trainmetric0,SIGMA))
    plt.plot(gaussian_filter1d(trainmetric1,SIGMA))
    plt.grid()

    plt.subplot(345)
    plt.ylabel('Val Patches loss')
    plt.plot(gaussian_filter1d(training_history['val_patches_loss'], VAL_SIGMA))
    plt.grid()

    
    valmetrics = np.array(training_history['val_patches_metrics'])
    valmetric0 = [x[0] for x in valmetrics]
    valmetric1 = [x[1] for x in valmetrics]
    
    plt.subplot(346)
    plt.ylabel('Val Patches Dice')
    plt.plot(valmetric0, '.-')
    plt.grid()

    plt.subplot(347)
    plt.plot(valmetric1, '.-', color='orange')
    plt.grid()

    plt.subplot(348)
    plt.plot((np.array(valmetric0) + np.array(valmetric1))/2, 'k.-')
    plt.legend(['Mean dice',])
    plt.grid()

    plt.subplot(349)
    plt.ylabel('Val Full Slice loss')
    plt.plot(gaussian_filter1d(training_history['val_full_slice_loss'], VAL_SIGMA))
    plt.grid()

    
    valmetrics = np.array(training_history['val_full_slice_metrics'])
    
    plt.subplot(3,4,10)
    plt.ylabel('Val Full Slice Dice')
    plt.plot(valmetrics[:,0], '.-')
    plt.grid()

    plt.subplot(3,4,11)
    plt.plot(valmetrics[:,1], '.-', color='orange')
    plt.grid()

    plt.subplot(3,4,12)
    plt.plot(np.mean(valmetrics,1), 'k.-')

    plt.legend(['Mean dice',])
    plt.grid()
    #plt.tight_layout()
    if SAVE:
        plt.savefig(OUTPUT + NAME)






def load_data(train, INDEX_LIST, MODE, N_PATCHES, INPUT):
    X = []
    Y = []
    for INDEX in INDEX_LIST:
        IMG = np.load(TEST_DATA_PATH + train[INDEX] + '_DATA.npy')
        LABEL = np.load(TEST_DATA_PATH + train[INDEX] +  '_VOTE_LABEL_V1_V2_V3_V4_Model_Baseline_F1_F2_F3_F4.npy')
        LABEL = np.mean(LABEL[6:],0)  # Take the mean across the segmentations (no consensus yet)
        if MODE == 'VOTE':
            LABEL[LABEL > 0.25] = 1
            LABEL[LABEL < 1] = 0
        elif MODE == 'AND':
            LABEL[LABEL > 0.5] = 1
            LABEL[LABEL < 1] = 0
        elif MODE == 'OR':
            LABEL[LABEL > 0] = 1
            LABEL[LABEL < 1] = 0
            
        if INPUT < IMG.shape[2]:
            lesion = np.argwhere(LABEL > 0)
            lesion = [x for x in lesion if INPUT/2 < x[0] < 507-INPUT/2-1 and INPUT/2 < x[1] < 507-INPUT/2-1 ]            
            bkg = np.argwhere(LABEL == 0)
            bkg = [x for x in bkg if INPUT/2 < x[0] < 507-INPUT/2-1 and INPUT/2 < x[1] < 507-INPUT/2-1 ]            
            coords1 = [ list(lesion[ii]) for ii in np.random.choice(range(len(lesion)), int(N_PATCHES/4))]
            coords2 = [ list(bkg[ii]) for ii in np.random.choice(range(len(bkg)), int(3*N_PATCHES/4))]
            coords = coords1
            coords.extend(coords2)
            for coord in coords:
                X.append(IMG[0,:,coord[0] - INPUT/2 : coord[0] + INPUT/2 + 1,
                                 coord[1] - INPUT/2 : coord[1] + INPUT/2 + 1  ,:])
                Y.append(LABEL[coord[0] - INPUT/2 : coord[0] + INPUT/2 + 1, 
                               coord[1] - INPUT/2 : coord[1] + INPUT/2 + 1  ])  
        elif INPUT == IMG.shape[2]:
            X.append(IMG[0])
            Y.append(LABEL)
    X = np.array(X)
    Y = np.array(Y)
    Y = tf.keras.utils.to_categorical(Y)
    Y = np.reshape(Y, ((Y.shape[0],) + (1,) + Y.shape[1:]))
    return X, Y

def load_data_test(train, INDEX_LIST, MODE, INPUT):
    X = []
    Y = []
    scanIDs = []
    for INDEX in INDEX_LIST:
        scanIDs.append(train[INDEX])
        IMG = np.load(TEST_DATA_PATH + train[INDEX] + '_DATA.npy')
        LABEL = np.load(TEST_DATA_PATH + train[INDEX] +  '_VOTE_LABEL_V1_V2_V3_V4_Model_Baseline_F1_F2_F3_F4.npy')
        X.append(IMG[0])
        Y.append(LABEL)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y, scanIDs


def train_model_crossval_test(model_name, model, INPUT_SHAPE, INPUT_SHAPE_VAL, MODEL_REDUCTION, BATCH_SIZE, FULL_SLICE_REDUCTION, history_callback, history_dict, X_train, Y_train, X_val_full_slice, Y_val_full_slice, X_val_patches, Y_val_patches, SHUFFLE=False):
    print('------ {} -------'.format(model_name))    
    DATA_SHAPE = X_train.shape[-2]
    OUTPUT_SHAPE = INPUT_SHAPE - MODEL_REDUCTION    
    
    model.fit(X_train[:,:,DATA_SHAPE/2-INPUT_SHAPE/2:DATA_SHAPE/2+INPUT_SHAPE/2+1,
                          DATA_SHAPE/2-INPUT_SHAPE/2:DATA_SHAPE/2+INPUT_SHAPE/2+1,:], 
              Y_train[:,:,DATA_SHAPE/2-OUTPUT_SHAPE/2:DATA_SHAPE/2+OUTPUT_SHAPE/2+1,
                          DATA_SHAPE/2-OUTPUT_SHAPE/2:DATA_SHAPE/2+OUTPUT_SHAPE/2+1,:], 
              epochs = 1, batch_size=BATCH_SIZE, callbacks = [history_callback], shuffle=SHUFFLE)
    history_dict['loss'].extend(history_callback.losses)
    history_dict['metrics'].append(history_callback.metrics)

    INPUT_SHAPE = X_val_full_slice.shape[-2]-FULL_SLICE_REDUCTION
    OUTPUT_SHAPE = INPUT_SHAPE - MODEL_REDUCTION
    val = model.evaluate(X_val_full_slice[:,:,507/2-INPUT_SHAPE/2:507/2+INPUT_SHAPE/2+1,507/2-INPUT_SHAPE/2:507/2+INPUT_SHAPE/2+1,:], 
                         Y_val_full_slice[:,:,507/2-OUTPUT_SHAPE/2:507/2+OUTPUT_SHAPE/2+1,507/2-OUTPUT_SHAPE/2:507/2+OUTPUT_SHAPE/2+1,:], 
                         batch_size=1, verbose=True)
    print(val)
    history_dict['val_full_slice_loss'].append(val[0])
    history_dict['val_full_slice_metrics'].append(val[2:])   

    DATA_SHAPE = X_val_patches.shape[-2]
    OUTPUT_SHAPE = INPUT_SHAPE_VAL - MODEL_REDUCTION
    val = model.evaluate(X_val_patches[:,:,DATA_SHAPE/2-INPUT_SHAPE_VAL/2:DATA_SHAPE/2+INPUT_SHAPE_VAL/2+1,
                                           DATA_SHAPE/2-INPUT_SHAPE_VAL/2:DATA_SHAPE/2+INPUT_SHAPE_VAL/2+1,:],  
                         Y_val_patches[:,:,DATA_SHAPE/2-OUTPUT_SHAPE/2:DATA_SHAPE/2+OUTPUT_SHAPE/2+1,
                                           DATA_SHAPE/2-OUTPUT_SHAPE/2:DATA_SHAPE/2+OUTPUT_SHAPE/2+1,:], 
                         batch_size=BATCH_SIZE, verbose=True)
    print(val)
    history_dict['val_patches_loss'].append(val[0])
    history_dict['val_patches_metrics'].append(val[2:])  
        

def segment_whole(X, model):
    output_dpatch = [0,205,205]
    shape = [0,507,507]
    
    if shape[1] == output_dpatch[1]:
        yend = output_dpatch[1]
    else:
        yend = output_dpatch[1] * int(round(float(shape[1])/output_dpatch[1] + 0.5)) 
    if shape[2] == output_dpatch[2]:
        zend = output_dpatch[2]
    else:           
        zend = output_dpatch[2] * int(round(float(shape[2])/output_dpatch[2] + 0.5))
    
    voxelCoordinates = []
    for y in range(output_dpatch[1]/2,yend,output_dpatch[1]):
        for z in range(output_dpatch[2]/2,zend,output_dpatch[2]):
            voxelCoordinates.append([y,z])  
    
    XX = np.pad(X[:,:,:,:,:], ((0,0),(0,0),(150,150),(150,150),(0,0)), 'reflect')
    
    pred = np.empty((700,700))
    for coords in voxelCoordinates:
        y1 = model.predict(XX[:,:,coords[0]-121+150:coords[0]+122+150, 
                                  coords[1]-121+150:coords[1]+122+150,:])
        pred[coords[0]-102:coords[0]+103,
             coords[1]-102:coords[1]+103] = y1[0,0,:,:,1]
    
    return pred[:507,:507]


def generalized_dice_completeImages(img1,img2, classes_GT=2):
    assert img1.shape == img2.shape, 'Images of different size!'
    #assert (np.unique(img1) == np.unique(img2)).all(), 'Images have different classes!'
    classes = np.array(np.unique(img1), dtype='int8')   
    if len(classes) < len(np.array(np.unique(img2), dtype='int8')):
      classes = np.array(np.unique(img2), dtype='int8')   
    dice = []
    if len(classes) == 1:
        print('Both images empty')
        return((0.0,[0.0,0.0]))
    for i in classes:
        dice.append(2*np.sum(np.multiply(img1==i,img2==i))/float(np.sum(img1==i)+np.sum(img2==i)))   
    return np.sum(dice)/len(classes), [x for x in dice]


def points_in_circle_np(radius, x0=0, y0=0, ):
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= radius**2)
    # x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation
    points = []
    
    for x, y in zip(x_[x], y_[y]):
        points.append([x, y])
    return points


def match_connected_component(data, GT_slice):

  structure = np.ones((3,3), dtype=np.int)
                
  labeled, ncomponents = label(data, structure)  
  sizes = {}
  for i in range(1,ncomponents+1):
    sizes[i] = len(np.argwhere(labeled == i))
  
  components = sizes.values()
  components.sort(reverse=1)

  # Remove components that dont overlap with GT
  for key in sizes:
    temp = np.zeros(GT_slice.shape)
    temp[labeled == key] = 1
    #plt.imshow(temp + GT_slice*4)
    if not (temp * GT_slice).any() :
      np.unique(labeled)
      labeled[labeled == key] = 0
      
  labeled[labeled > 0] = 1    
  return labeled      

def HalfMax_MaxSeed(pred, MAX_THRESHOLD_FACTOR, SEED=True):      
    MAX = np.max(pred)
    THRESHOLD = MAX*float(MAX_THRESHOLD_FACTOR)
    if np.min(pred) > THRESHOLD:
        return np.zeros(pred.shape)
    #print('MAX_THRESHOLD_FACTOR: {}'.format(THRESHOLD))
    MAX = MAX * 0.95 
    Max_points = np.argwhere(pred[10:-10,10:-10] > MAX)    # avoid border of size point-radius      

    Dummy = np.zeros(pred.shape)
    for center in Max_points:                                
        pp = points_in_circle_np(radius=10, x0=center[0], y0=center[1])
        for point in pp:
            Dummy[point[0],point[1]] += 1

    mach_thr = np.array(pred)        
    mach_thr[pred < THRESHOLD] = 0
    mach_thr[pred >= THRESHOLD] = 1  
    
    if SEED:
        mach_thr = match_connected_component(mach_thr, Dummy)    
    
    return mach_thr
#%%

PATH_MODELS = '/home/deeperthought/Projects/MultiPriors_MSKCC/models/Patch_training/'

INPUT = 107
N_PATCHES = 2

TEST_DATA_PATH = '/media/deeperthought/DATA2/DATA/Test_Malignants_Full_Slice_Patches/'

data = os.listdir(TEST_DATA_PATH)    
data = [x.split('_VOTE')[0] for x in data if 'VOTE' in x]


remove = ['MSKCC_16-328_1_04021_20060805_r',
            'MSKCC_16-328_1_00804_20021013_l',
            'MSKCC_16-328_1_00590_20020809_r',
            'MSKCC_16-328_1_02471_20070814_r',
            'MSKCC_16-328_1_00565_20020702_r',
            'MSKCC_16-328_1_03718_20060421_l']


data = [x for x in data if x not in remove]

N = len(data)/2


# Two cross-val splits
# three methods
# --> 6 combinations. 6 models, 6 results.

from Unet_3D_Class import Generalised_dice_coef_multilabel2, dice_coef_multilabel_bin0, dice_coef_multilabel_bin1
from tensorflow.keras.optimizers import Adam

OUTPUT_PATH = '/home/deeperthought/Projects/MultiPriors_MSKCC/models/Patch_training/best_models/CrossVal_Test/'
     
for MODE in [ 'AND', 'VOTE', 'OR']:

    half1, half2 = data[:N], data[N:]
    
    for datasplit in [0,1]:
        
        # Instead of loading model, make model, with regularization, and load weights.
        with tf.device('GPU:0'):
            unet_v4 = load_model(PATH_MODELS + 'Unet_v4_5M_batch16_Patch_19_75_75_3_model.h5',my_custom_objects)
        
        for layer in unet_v4.layers[:-2]:
            layer.trainable = False

        unet_v4.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=1e-3), 
                      metrics=['acc', dice_coef_multilabel_bin0, dice_coef_multilabel_bin1])

        with tf.device('GPU:1'):
            best_model = tf.keras.models.clone_model(unet_v4)            
#        for layer in unet_v4.layers:
#            print layer.trainable
        
        # Split data
        if datasplit == 0:
            train = half1
            test = half2
        else:
            train = half2
            test = half1
            
        train, val = train[:-33], train[-33:]
        
        history_v4 = LossHistory_multiDice2() 
        training_history_v4 = {'loss':[], 'metrics':[], 
                               'val_full_slice_loss':[], 'val_full_slice_metrics':[],
                               'val_patches_loss':[], 'val_patches_metrics':[]}
        
        INDEX_LIST_VAL = np.arange(0, len(val))
        X_val, Y_val = load_data(val, INDEX_LIST_VAL, MODE, 16, INPUT)
        X_val_full_slice, Y_val_full_slice = load_data(val, INDEX_LIST_VAL, MODE, 1, 507)
        
        N_SUBJECTS = 32
        START = 0

        val_model('unet_v4', unet_v4, 75, 38, 12, 160+32, history_v4, training_history_v4, X_val_full_slice, Y_val_full_slice, X_val, Y_val)

        MAX = training_history_v4['val_full_slice_metrics'][0][1]
        print('Value to beat: {}'.format(MAX))
        for jjj in range(50):
            INDEX_LIST = np.arange(START, START + N_SUBJECTS) % len(train)
            START += N_SUBJECTS        
            X_train, Y_train = load_data(train, INDEX_LIST, MODE, 16, INPUT)
            
            train_model('unet_v4', unet_v4,  75, 75, 38, 12, 160+32, history_v4, training_history_v4, X_train, Y_train, 
                        X_val_full_slice, Y_val_full_slice, X_val, Y_val, SHUFFLE=True)
        
            LAST = training_history_v4['val_full_slice_metrics'][-1][1]
            if LAST > MAX:
                print('Iter {} ### New max dice: {} vs old {}'.format(jjj, LAST, MAX))
                MAX = LAST
                ww = unet_v4.get_weights()
                best_model.set_weights(ww)
            
        
        SAVE_FLAG = 1
        SIGMA = 50
        VAL_SIGMA = 0.0001
                
        #np.save(OUTPUT_PATH + 'training_history_v4_5M_batch16_input75.npy', training_history_v4_1, allow_pickle=True)
        
        plot_training(training_history_v4, SIGMA, VAL_SIGMA, 'unet_v4_last_layer_{}_Split{}_LR1e3'.format(MODE, datasplit), SAVE_FLAG, OUTPUT_PATH)    

        best_model.save_weights(OUTPUT_PATH + '{}_{}_weights.h5'.format(MODE, datasplit))        
        best_model.save(OUTPUT_PATH + '{}_{}_model.h5'.format(MODE, datasplit))   

        MAX_THRESHOLD_FACTOR = 0.5
        M2 = {}
        df = pd.DataFrame(columns=['scanID', 'M_before_vs_GT1', 'M_before_vs_GT2', 'M_before_vs_GT3', 'M_before_vs_GT4', 'M_vs_GT1', 'M_vs_GT2', 'M_vs_GT3', 'M_vs_GT4'])
        X_test, Y_test, test_scanIDs = load_data_test(test, range(0, len(test)), MODE, 507)
        
        OUTPUT_PRED_PATH = '/home/deeperthought/Projects/MultiPriors_MSKCC/models/Patch_training/best_models/CrossVal_Test/RESULTS/'
        for iii in range(len(X_test)):
            print(iii)
            VOTE234 = Y_test[iii,0]
            VOTE134 = Y_test[iii,1]
            VOTE124 = Y_test[iii,2]
            VOTE123 = Y_test[iii,3]
            
            y_hat = segment_whole(X_test[iii:iii+1], unet_v4)
            np.save(OUTPUT_PRED_PATH + 'Model_before_{}_{}.npy'.format(test_scanIDs[iii], MODE), y_hat)
            
            pred_max = HalfMax_MaxSeed(y_hat, MAX_THRESHOLD_FACTOR)
            d1 = generalized_dice_completeImages(pred_max,VOTE234)[1][1]
            d2 = generalized_dice_completeImages(pred_max,VOTE134)[1][1]
            d3 = generalized_dice_completeImages(pred_max,VOTE124)[1][1]
            d4 = generalized_dice_completeImages(pred_max,VOTE123)[1][1]

            M2['scanID'] = test_scanIDs[iii]
            M2['M_before_vs_GT1'] = d1
            M2['M_before_vs_GT2'] = d2
            M2['M_before_vs_GT3'] = d3
            M2['M_before_vs_GT4'] = d4
            
            y_hat = segment_whole(X_test[iii:iii+1], best_model)
            np.save(OUTPUT_PRED_PATH + 'Model_after_{}_{}.npy'.format(test_scanIDs[iii], MODE), y_hat)

            pred_max = HalfMax_MaxSeed(y_hat, MAX_THRESHOLD_FACTOR)
            d1 = generalized_dice_completeImages(pred_max,VOTE234)[1][1]
            d2 = generalized_dice_completeImages(pred_max,VOTE134)[1][1]
            d3 = generalized_dice_completeImages(pred_max,VOTE124)[1][1]
            d4 = generalized_dice_completeImages(pred_max,VOTE123)[1][1]
            
            M2['M_vs_GT1'] = d1
            M2['M_vs_GT2'] = d2
            M2['M_vs_GT3'] = d3
            M2['M_vs_GT4'] = d4
         
            df = df.append(M2, ignore_index=True)

        df.to_csv(OUTPUT_PRED_PATH + '{}_DataSplit{}.csv'.format(MODE, datasplit))
