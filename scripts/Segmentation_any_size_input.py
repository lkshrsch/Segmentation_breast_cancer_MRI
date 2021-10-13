#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:47:01 2020

@author: deeperthought
"""

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="2"
tf.keras.backend.set_session(tf.Session(config=config))

import os
import numpy as np
import nibabel as nib
from keras.models import load_model
from keras.models import Model
import sys
sys.path.append('/home/deeperthought/Projects/MultiPriors_MSKCC/scripts/')
import matplotlib.pyplot as plt
from lib import *
from skimage.transform import resize
from MultiPriors_Models_Collection import Generalised_dice_coef_multilabel2, dice_coef_multilabel_bin0,dice_coef_multilabel_bin1
import scipy.ndimage as ndimage

#%%    USER INPUT
     
PATH_TO_MODEL = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/UNet_3D_v4_MSKCC_HumanSegmentation_configFile_UNet_3D_v4_HumanPerformance_percentileNorm_2020-05-12_1911/models/best_model.h5'

WD = '/home/deeperthought/Projects/MultiPriors_MSKCC'


/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_HumanPerformance_Segmentation/val_labels.txt
/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_HumanPerformance_Segmentation/val_slope1.txt
/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_HumanPerformance_Segmentation/val_slope2.txt
/home/deeperthought/Projects/MultiPriors_MSKCC/CV_folds/CV_HumanPerformance_Segmentation/val_t1post.txt

segmentChannels = [WD + '/CV_folds/CV_HumanPerformance_Segmentation/val_t1post.txt',
            		 WD + '/CV_folds/CV_HumanPerformance_Segmentation/val_slope1.txt',
            		 WD + '/CV_folds/CV_HumanPerformance_Segmentation/val_slope2.txt']

VISUALIZE_FEATURES = False

OUTPUT_FOLDER = PATH_TO_MODEL..split('models')[0] + '/SEGMENTATIONS_TEST/'

SEG_PATCH = (27, 99, 99)



INDEXES = [0]

#%%  Define functions


#channel = testChannels[0]
#subjectIndex = 0
#subject_channel_voxelCoordinates = voxelCoordinates
#output_shape = shape
#dpatch = segmentation_dpatch

def extractImagePatch_parallelization(channel, subjectIndex, subject_channel_voxelCoordinates, output_shape, dpatch, percentile_normalization, preprocess_image_data=True,fullSegmentationPhase=False):   
    subject_channel = getSubjectsToSample(channel, [subjectIndex])
    n_patches = len(subject_channel_voxelCoordinates)
    subject = str(subject_channel[0])[:-1]
    proxy_img = nib.load(subject)            
    img_data = np.array(proxy_img.get_data(),dtype='float32')
    
    # SWAP #
    img_data = np.swapaxes(img_data, 0,2)
    
    if preprocess_image_data:   
      if np.array(img_data.shape != output_shape).any():
        print('Resizing training data: \nInput_shape = {}, \nOutput_shape = {}. \nSubject = {}'.format(img_data.shape, output_shape, subject))
        img_data = resize(img_data, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')
      if np.any(np.isnan(img_data)):
        print('Nans found in scan {}'.format(subject))
        print('Nans replace by value: {}'.format(np.nanmin(img_data)))
        img_data[np.isnan(img_data)] = np.nanmin(img_data)
      
      if percentile_normalization:
        img_data = img_data/np.percentile(img_data, 90)
      else:
        img_data = normalizeMRI(img_data)      
      if not np.isfinite(img_data).all():
        print('Normalization: Nans found in scan {}'.format(subject))
        print('Nans replace by value: {}'.format(np.nanmin(img_data)))
        img_data[ ~ np.isfinite(img_data)] = np.nanmin(img_data)
        
    if fullSegmentationPhase:      
        if np.max(dpatch) > 200:  # This is the case with the full-image U-Net_v0. If we pad too big, this takes a lot of time and unnecessary resources.
            padding_border = 10#np.max(dpatch)#np.max(dpatch)/2 + 10#550
        else:
            padding_border = np.max(dpatch)
    else:
        padding_border = np.max(dpatch)/2 + 10
    # Padding needs to be larger than dpatch/2. During training all patches are centered within the image so here its enough.
    # But during testing, we need to sample over all center patches, which means that we can go outside the original image boundaries.
    # Example: image size = 7, center patch size = 3
    # Image : [0123456] center-patches: [012][345][678] . Last patch was centered on 7, just to capture the 6 on the border. 
    #print('Padding image..')
    img_data_padded = np.pad(img_data, padding_border,'reflect')    
    
    vol = np.zeros((n_patches,dpatch[0],dpatch[1],dpatch[2]),dtype='float32') 
    for j in range(n_patches):      
        D1,D2,D3 = subject_channel_voxelCoordinates[j]           
        D1 = D1 + padding_border#dpatch[0]/2
        D2 = D2 + padding_border#dpatch[1]/2
        D3 = D3 + padding_border#dpatch[2]/2
        try:
          vol[j,:,:,:] = img_data_padded[D1-(dpatch[0]/2):D1+(dpatch[0]/2)+dpatch[0]%2,
                                         D2-(dpatch[1]/2):D2+(dpatch[1]/2)+dpatch[1]%2,
                                         D3-(dpatch[2]/2):D3+(dpatch[2]/2)+dpatch[2]%2]
        except:
          print('Failed to extract image data into shape... This is: \n{}, \nimg_data_padded.shape = {}, \nCoords = {}, \nCoords+Padding = {}'.format(subject_channel,img_data_padded.shape, subject_channel_voxelCoordinates[j] , [D1,D2,D3] ))
          sys.exit(0)
    proxy_img.uncache()
    del img_data
    del img_data_padded
    return vol


#TPM_channel=[]
#model=model
#testChannels=segmentChannels
#testLabels=''
#output_classes=2 
#subjectIndex=0
#segmentation_dpatch=[27,139,139]
#size_minibatches=16
#use_coordinates=False
#percentile_normalization=True
#model_patch_reduction=[18, 38, 38]
             
def get_features(TPM_channel, model, testChannels, testLabels, output_classes, subjectIndex, segmentation_dpatch, size_minibatches, use_coordinates, percentile_normalization, model_patch_reduction):    
   
    output_dpatch = segmentation_dpatch[0] - model_patch_reduction[0], segmentation_dpatch[1] - model_patch_reduction[1], segmentation_dpatch[2] - model_patch_reduction[2]     
    subjectIndex = [subjectIndex]
    num_channels = len(testChannels)
    firstChannelFile = open(testChannels[0],"r")   
    ch = firstChannelFile.readlines()
    subjectGTchannel = ch[subjectIndex[0]][:-1]
    subID = subjectGTchannel.split('/')[-2] + '_' + subjectGTchannel.split('/')[-1].split('.nii')[0]
    print('SEGMENTATION : Segmenting subject: ' + str(subID))  

    firstChannelFile.close()      
    proxy_img = nib.load(subjectGTchannel)
    shape = proxy_img.shape
    affine = proxy_img.affine      
    res = proxy_img.header['pixdim'][1:4]
    
    
    # SWAP AXIS #
    res[0] = res[2]
    res[2] = res[1]  
    shape = (shape[2], shape[1], shape[0])


    if res[1] > 0.6:    
      target_res = [res[0],res[1]/2.,res[2]/2.]
      shape = [int(x) for x in np.floor([float(s)*r1/r2 for s,r1,r2 in zip(shape, res, target_res)])]
    else:
      target_res = res
      shape=list(shape)
    
          
    print('SEGMENTATION : Sampling data..')  
    TPM_patches, labels, voxelCoordinates, spatial_coordinates, shape = sampleTestData(TPM_channel, testChannels, testLabels, subjectIndex, output_classes, 
                                                                                       output_dpatch, shape, use_coordinates)    
    affine = np.diag(list(target_res) + [0])        
    n_minibatches = 0 # min(0,len(voxelCoordinates)/size_minibatches) 
    total_number_of_patches = (len(voxelCoordinates)-n_minibatches*size_minibatches)  
    
    #########################################################################
    print('SEGMENTATION : Extracting {} image patches..'.format(total_number_of_patches))
    patches = np.zeros((total_number_of_patches,segmentation_dpatch[0],segmentation_dpatch[1],segmentation_dpatch[2],num_channels),dtype='float32')
    for i in range(len(testChannels)):
        print('Extracting patches from {}'.format(testChannels[i]))
        patches[:,:,:,:,i] = extractImagePatch_parallelization(testChannels[i], subjectIndex[0], voxelCoordinates, shape, segmentation_dpatch, percentile_normalization, fullSegmentationPhase=True)    

    print('SEGMENTATION : Finished sampling data.')
    INPUT_DATA = []   
    INPUT_DATA.append(patches)

    if len(spatial_coordinates) > 0:
        INPUT_DATA.append(spatial_coordinates)            

    print("SEGMENTATION : Finished preprocessing data for segmentation.")
    #########################################################################
    
    segmentation = model.predict(INPUT_DATA, verbose=1, batch_size=8)
    #return features, segmentation       

    #shape.append(2)
    output2 = np.ones(shape, dtype=np.float32)  # same size as input head, start index for segmentation start at 26,26,26, rest filled with zeros....
    i = 0
    for x,y,z in voxelCoordinates:
        patch_shape = output2[x-output_dpatch[0]/2:min(x+(output_dpatch[0]/2+output_dpatch[0]%2), shape[0]),
                           y-output_dpatch[1]/2:min(y+(output_dpatch[1]/2+output_dpatch[1]%2), shape[1]),
                           z-output_dpatch[2]/2:min(z+(output_dpatch[2]/2+output_dpatch[2]%2), shape[2])].shape
        #print(np.array(indexes[i])[0:patch_shape[0], 0:patch_shape[1],0:patch_shape[2]])
        output2[x-output_dpatch[0]/2:min(x+(output_dpatch[0]/2+output_dpatch[0]%2), shape[0]),
             y-output_dpatch[1]/2:min(y+(output_dpatch[1]/2+output_dpatch[1]%2), shape[1]),
             z-output_dpatch[2]/2:min(z+(output_dpatch[2]/2+output_dpatch[2]%2), shape[2])] = np.array(segmentation[i])[0:patch_shape[0], 
                                                                                                                      0:patch_shape[1],
                                                                                                                      0:patch_shape[2],
                                                                                                                      1]
        i = i+1

    
    img = nib.Nifti1Image(output2, affine)
    return img, subID



#%%    Load Model, get intermediate layer, get features
my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
                                 'dice_coef_multilabel_bin0':dice_coef_multilabel_bin0,
                                 'dice_coef_multilabel_bin1':dice_coef_multilabel_bin1}

model = load_model(PATH_TO_MODEL, custom_objects = my_custom_objects)

#from Unet_3D_Class import UNet_v4
#new_model = UNet_v4(input_shape=(None,None,None,3), pool_size=(2, 2, 2), n_labels=2, initial_learning_rate=0.00001, deconvolution=True,
#      depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, 
#      batch_normalization=True, activation_name="softmax", bilinear_upsampling=True)
#
#
#for i in range(len(model.layers)):
#    print(i)
#    new_model.layers[i].set_weights(model.layers[i].get_weights())
#
#
#new_model.save('/home/deeperthought/Projects/MultiPriors_MSKCC/models/fastSegmenter.h5')

SUBJECT_INDEX = 0

for SUBJECT_INDEX in INDEXES:
    subjectIndex = [SUBJECT_INDEX]
    num_channels = len(segmentChannels)
    firstChannelFile = open(segmentChannels[0],"r")   
    ch = firstChannelFile.readlines()
    subjectGTchannel = ch[subjectIndex[0]][:-1]
    subID = subjectGTchannel.split('/')[-2] + '_' + subjectGTchannel.split('/')[-1].split('.nii')[0]
    print('SEGMENTATION : Segmenting subject: ' + str(subID))  
    scanID = subID.replace('_T1','').replace('_02_01','_segmentation')
    if os.path.exists(OUTPUT_FOLDER + 'ROI/' + '{}'.format(scanID) + '.npy'):
        print('Already done. Skip.')
        continue
    
    img, subID = get_features(TPM_channel=[], model=new_model, testChannels=segmentChannels, testLabels='', output_classes=2, subjectIndex=SUBJECT_INDEX, segmentation_dpatch=SEG_PATCH,#[19, 75, 75],
                 size_minibatches=8, use_coordinates=False, percentile_normalization=True, model_patch_reduction=[18, 38, 38])
    

    nib.save(img, OUTPUT_FOLDER + '/' + scanID)



 
    

