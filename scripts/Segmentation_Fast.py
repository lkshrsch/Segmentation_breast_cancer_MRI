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
import pandas as pd
from keras.models import load_model
from keras.models import Model
import sys
sys.path.append('/home/deeperthought/Projects/MultiPriors_MSKCC/scripts/')
import matplotlib.pyplot as plt
import lib 
from skimage.transform import resize
from MultiPriors_Models_Collection import Generalised_dice_coef_multilabel2, dice_coef_multilabel_bin0,dice_coef_multilabel_bin1
import scipy.ndimage as ndimage
import json
from keras.utils import to_categorical
from scipy.interpolate import interp1d
#%%    USER INPUT
     
PATH_TO_MODEL = '/home/deeperthought/Projects/MultiPriors_MSKCC/training_sessions/UNet_3D_v4_MSKCC_HumanSegmentation_configFile_UNet_3D_v4_HumanPerformance_percentileNorm_2020-05-14_1614/models/last_model.h5'

##--------- Sagittal ----------------
#
#segmentChannels1 = ['/media/deeperthought/alignedNii-Nov2019/MSKCC_16-328_1_14150_20100408/T1_right_01_01.nii.gz',
#                   '/media/deeperthought/alignedNii-Nov2019/MSKCC_16-328_1_14150_20100408/T1_right_02_01.nii.gz',
#                   '/media/deeperthought/alignedNii-Nov2019/MSKCC_16-328_1_14150_20100408/T1_right_slope1.nii.gz',
#                   '/media/deeperthought/alignedNii-Nov2019/MSKCC_16-328_1_14150_20100408/T1_right_slope2.nii.gz']
#
#segmentLabels1 = '/home/deeperthought/Projects/MSKCC/Segmenter_software/F3_segmentations_Jan2020/Batch_Feb_2020/Isaac/segmentation/MSKCC_16-328_1_14150_20100408_r.nii.gz'
#
##----------- AXIAL ------------------
#segmentChannels2 = ['/media/deeperthought/AXIAL/MRI/Malignants/RIA_19-093_000_00038_20161114/T1_axial_01_01.nii',
#                    '/media/deeperthought/AXIAL/MRI/Malignants/RIA_19-093_000_00038_20161114/T1_axial_02_01.nii',
#                    '/media/deeperthought/AXIAL/MRI/Malignants/RIA_19-093_000_00038_20161114/T1_axial_slope1.nii',
#                    '/media/deeperthought/AXIAL/MRI/Malignants/RIA_19-093_000_00038_20161114/T1_axial_slope2.nii']
#
#segmentLabels2 = '/media/deeperthought/AXIAL/Segmentations/RIA_19-093_000_00038_20161114_seg.nii'
#
##---------- AXIALS ROTATED ------------
#segmentChannels3 = ['/media/deeperthought/AXIAL/MRI/Malignants/RIA_19-093_000_00038_20161114_rotated/T1_axial_01_01_rotated.nii',
#                '/media/deeperthought/AXIAL/MRI/Malignants/RIA_19-093_000_00038_20161114_rotated/T1_axial_02_01_rotated.nii',
#                '/media/deeperthought/AXIAL/MRI/Malignants/RIA_19-093_000_00038_20161114_rotated/T1_axial_slope1_rotated.nii',
#                '/media/deeperthought/AXIAL/MRI/Malignants/RIA_19-093_000_00038_20161114_rotated/T1_axial_slope2_rotated.nii']
#
#segmentLabels3 = '/media/deeperthought/AXIAL/Segmentations/RIA_19-093_000_00038_20161114_seg_rotated.nii'

#---------- AXIALS ROTATED ------------
segmentChannels = ['/media/deeperthought/AXIAL/MRI/Malignants/RIA_19-093_000_00038_20161114_rotated/T1_axial_02_01_rotated_resolution.nii',
                   '/media/deeperthought/AXIAL/MRI/Malignants/RIA_19-093_000_00038_20161114_rotated/T1_axial_slope1_rotated_resolution.nii',
                   '/media/deeperthought/AXIAL/MRI/Malignants/RIA_19-093_000_00038_20161114_rotated/T1_axial_slope2_rotated_resolution.nii']

segmentLabels = '/media/deeperthought/AXIAL/Segmentations/RIA_19-093_000_00038_20161114_seg_rotated_resolution.nii'

##----------- AXIALS ROTATED AND RESCALED ------
#segmentChannels5 = ['/media/deeperthought/AXIAL/MRI/Malignants/RIA_19-093_000_00038_20161114_rotated/T1_axial_01_01_rotated_resolution_rescaled.nii',
#                '/media/deeperthought/AXIAL/MRI/Malignants/RIA_19-093_000_00038_20161114_rotated/T1_axial_02_01_rotated_resolution_rescaled.nii',
#                '/media/deeperthought/AXIAL/MRI/Malignants/RIA_19-093_000_00038_20161114_rotated/T1_axial_slope1_rotated_resolution_rescaled.nii',
#                '/media/deeperthought/AXIAL/MRI/Malignants/RIA_19-093_000_00038_20161114_rotated/T1_axial_slope2_rotated_resolution_rescaled.nii']
#
#
#segmentLabels5 = '/media/deeperthought/AXIAL/Segmentations/RIA_19-093_000_00038_20161114_seg_rotated_resolution.nii'
#


OUTPUT_PATH = PATH_TO_MODEL.split('models')[0] + '/SEGMENTATIONS_TEST/'

SEG_PATCH = (27, 99, 99)

intensity_normalization_method = 2  


#%%  Advanced parameters

model_patch_reduction = [18, 38, 38]

penalty_MATRIX = np.array([[ 1,  0],
			   [ 0,  1]], dtype='float32')
TPM_channel = []
output_classes = 2
segmentation_dpatch = SEG_PATCH
size_test_minibatches =  8

size_minibatches = 8
subjectIndex = 0
use_coordinates= False

model_patch_reduction = model_patch_reduction
model_crop = 0
resolution = 'high'

dice_compare = False


#%%

def generalized_dice_completeImages(img1,img2):
    assert img1.shape == img2.shape, 'Images of different size!'
    #assert (np.unique(img1) == np.unique(img2)).all(), 'Images have different classes!'
    classes = np.array(np.unique(img1), dtype='int8')   
    if len(classes) < len(np.array(np.unique(img2), dtype='int8')   ):
      classes = np.array(np.unique(img2), dtype='int8')   
    dice = []
    for i in classes:
        dice.append(2*np.sum(np.multiply(img1==i,img2==i))/float(np.sum(img1==i)+np.sum(img2==i)))   
    return np.sum(dice)/len(classes), [round(x,2) for x in dice]


def extractLabels(segmentLabels, voxelCoordinates, output_dpatch, shape):
    labels = []       
    proxy_label = nib.load(segmentLabels)
    label_data = np.array(proxy_label.get_data(),dtype='int8')
    label_data = resize(label_data, shape, order=0, preserve_range=True, anti_aliasing=True)        
    label_padded = np.pad(label_data,((0,60),(0,100),(0,100)),'constant')  # need to pad for segmentation with huge patches that go outside (only the end - ascending coordinates) boundaries. Scale stays the same, as the origin is not modified. 
    if np.sum(label_data) == 0:
      for j in range(len(voxelCoordinates)):
        labels.append(np.zeros((output_dpatch[0],output_dpatch[1],output_dpatch[2]),dtype='int8'))
    else:
      for j in range(len(voxelCoordinates)):
        D1,D2,D3 = voxelCoordinates[j]
        labels.append(label_padded[D1-output_dpatch[0]/2:D1+(output_dpatch[0]/2)+output_dpatch[0]%2,
                                   D2-output_dpatch[1]/2:D2+(output_dpatch[1]/2)+output_dpatch[1]%2,
                                   D3-output_dpatch[2]/2:D3+(output_dpatch[2]/2)+output_dpatch[2]%2])
    proxy_label.uncache()
    del label_data
    return labels


def sampleTestData(TPM_channel, segmentLabels, output_classes, output_dpatch, shape, use_coordinates):
   
    xend = output_dpatch[0] * int(round(float(shape[0])/output_dpatch[0] + 0.5)) 
    if shape[1] == output_dpatch[1]:
        yend = output_dpatch[1]
    else:
        yend = output_dpatch[1] * int(round(float(shape[1])/output_dpatch[1] + 0.5)) 
    if shape[2] == output_dpatch[2]:
        zend = output_dpatch[2]
    else:           
        zend = output_dpatch[2] * int(round(float(shape[2])/output_dpatch[2] + 0.5))
    voxelCoordinates = []
    # Remember in python the end is not included! Last voxel will be the prior-to-last in list.
    # It is ok if the center voxel is outside the image, PROPER padding will take care of that (if outside the image size, then we need larger than dpatch/2 padding)
    
    for x in range(output_dpatch[0]/2,xend,output_dpatch[0]): 
        for y in range(output_dpatch[1]/2,yend,output_dpatch[1]):
            for z in range(output_dpatch[2]/2,zend,output_dpatch[2]):
                voxelCoordinates.append([x,y,z])
    
    if len(TPM_channel) > 0:
      TPM_patches = lib.extract_TPM_patches(TPM_channel, voxelCoordinates, output_dpatch, [shape])
    else:
      TPM_patches = []
    if len(segmentLabels) > 0:
      labels = np.array(extractLabels(segmentLabels, voxelCoordinates, output_dpatch,shape))
      labels = to_categorical(labels.astype(int),output_classes)
    else:
      labels = []
    if use_coordinates:
      spatial_coordinates = lib.extractCoordinates([shape], [voxelCoordinates], output_dpatch) 
    else:
      spatial_coordinates = []
    #print("Finished extracting " + str(n_patches) + " patches, from "  + str(n_subjects) + " subjects and " + str(num_channels) + " channels. Timing: " + str(round(end-start,2)) + "s")
    return TPM_patches, labels, voxelCoordinates, spatial_coordinates, shape        



def extractImagePatch_parallelization(segmentChannels, voxelCoordinates, output_shape, dpatch, 
                                      intensity_normalization_method, preprocess_image_data=True,fullSegmentationPhase=False):   

    n_patches = len(voxelCoordinates)
    p95 = 0
    
#    if intensity_normalization_method == 2:
#        T1_pre_nii_path = segmentChannels[0]
#        print('Getting p95 from {}'.format(T1_pre_nii_path))
#        #p95 = np.percentile(nib.load(T1_pre_nii_path).get_data(),95)
#        
#        segmentChannels = segmentChannels[1:]

    subject_all_channels_img = []
       

    for mri_modality in segmentChannels:

        proxy_img = nib.load(mri_modality)            
        img_data = np.array(proxy_img.get_data(),dtype='float32')
        #img_data[img_data < 0] = 0
        proxy_img.uncache()
            
        if preprocess_image_data:   
        
            
          if np.array(img_data.shape != output_shape).any():
            #print('Resizing training data: \nInput_shape = {}, \nOutput_shape = {}. \nSubject = {}'.format(img_data.shape, output_shape, subject))
            img_data = resize(img_data, output_shape=output_shape, preserve_range=True, anti_aliasing=True, mode='reflect')            
            
          if np.any(np.isnan(img_data)):
            print('Nans found in scan')
            print('Nans replace by value: {}'.format(np.nanmin(img_data)))
            img_data[np.isnan(img_data)] = np.nanmin(img_data)
        
          
          if intensity_normalization_method == 1:
            #print('Normalizing intensities using histogram matching')
            # ALL THIS HARD-CODED PATHS NEED TO GO
            keys = pd.read_csv('/home/deeperthought/Projects/Intensity_Normalization_stuff/scanID_cluster_keys.csv')          
            if scanID in list(keys['scanID']):
                cluster = keys.loc[keys['scanID'] == scanID, 'cluster_ID'].values[0]            
            else:
                print('Scan not in dataframe!')
                sys.exit(0)
            
            #print('Loading LUT..')
            LUT_PATH = '/home/deeperthought/Projects/Intensity_Normalization_stuff/LUT/'
            LUT_arr = json.load(open(LUT_PATH + 'LUT_cluster_{}.json'.format(cluster), 'r'))
            x = [tup[0] for tup in LUT_arr]
            y = [tup[1] for tup in LUT_arr]
            
            # Extrapolation. I need to re-make the LUTs. With all 70k scans!!
            x[-1] = 50*x[-1]
            y[-1] = 50*y[-1]
            
            LUT_cluster = interp1d(x,y)
            #print('Normalizing scan..')  
            img_data = lib.normalize_using_LUT(img_data, LUT_cluster)
              
          elif intensity_normalization_method == 2:
            img_data = lib.percentile95_normalizeMRI(img_data, p95)
    
          elif intensity_normalization_method == 3:
            img_data = lib.normalizeMRI(img_data)      
            
          if not np.isfinite(img_data).all():
            print('Normalization: Nans found in scan')
            print('Nans replace by value: {}'.format(np.nanmin(img_data)))
            img_data[ ~ np.isfinite(img_data)] = np.nanmin(img_data)

        subject_all_channels_img.append(img_data)

    if intensity_normalization_method == 1:
        slope1 = subject_all_channels_img[1] - subject_all_channels_img[0]
        subject_all_channels_img.append(slope1)
        
    vol = np.zeros((n_patches,dpatch[0],dpatch[1],dpatch[2], len(segmentChannels)),dtype='float32') 

    for mri_modality in range(len(subject_all_channels_img)):
        img_data = subject_all_channels_img[mri_modality]
        if fullSegmentationPhase:      
            if np.max(dpatch) > 200:  # This is the case with the full-image U-Net_v0. If we pad too big, this takes a lot of time and unnecessary resources.
                padding_border = 10#np.max(dpatch)#np.max(dpatch)/2 + 10#550
            else:
                padding_border = np.max(dpatch)
        else:
            padding_border = np.max(dpatch)/2 + 10

        img_data_padded = np.pad(img_data, padding_border,'reflect')    
        
        for j in range(n_patches):      
            D1,D2,D3 = voxelCoordinates[j]           
            D1 = D1 + padding_border
            D2 = D2 + padding_border
            D3 = D3 + padding_border
            try:
              vol[j,:,:,:,mri_modality] = img_data_padded[D1-(dpatch[0]/2):D1+(dpatch[0]/2)+dpatch[0]%2,
                                             D2-(dpatch[1]/2):D2+(dpatch[1]/2)+dpatch[1]%2,
                                             D3-(dpatch[2]/2):D3+(dpatch[2]/2)+dpatch[2]%2]
            except:
              print('Failed to extract image data into shape... ')
              sys.exit(0)
              
              
    del img_data
    del img_data_padded
    return vol





def fullSegmentation_quick(penalty_MATRIX, resolution, TPM_channel, model, 
                     segmentChannels, segmentLabels, output_classes, segmentation_dpatch, size_minibatches, use_coordinates, 
                     intensity_normalization_method, model_patch_reduction, model_crop, using_breastMaskModel=False, MASK_BREAST=False,
                     using_Unet=False, using_unet_breastMask=False):    
    
    output_dpatch = segmentation_dpatch[0] - model_patch_reduction[0], segmentation_dpatch[1] - model_patch_reduction[1], segmentation_dpatch[2] - model_patch_reduction[2]   

    proxy_img = nib.load(segmentChannels[0])
    shape = proxy_img.shape
    affine = proxy_img.affine      
    res = proxy_img.header['pixdim'][1:4]

    if resolution == 'high':
        if res[1] > 0.6:    
          target_res = [res[0],res[1]/2.,res[2]/2.]
          shape = [int(x) for x in np.floor([float(s)*r1/r2 for s,r1,r2 in zip(shape, res, target_res)])]
        else:
          target_res = res
    elif resolution == 'low':
        if res[1] < 0.6:    
          target_res = [res[0],res[1]*2.,res[2]*2.]
          shape = [int(x) for x in np.floor([float(s)*r1/r2 for s,r1,r2 in zip(shape, res, target_res)])]
        else:
          target_res = res          
          
    print('SEGMENTATION : Sampling data..')  
    TPM_patches, labels, voxelCoordinates, spatial_coordinates, shape = sampleTestData(TPM_channel, segmentLabels, 
                                                                                       output_classes, 
                                                                                       output_dpatch, shape, use_coordinates)    
    affine = np.diag(list(target_res) + [0])        
    n_minibatches = 0 # min(0,len(voxelCoordinates)/size_minibatches) 
    total_number_of_patches = (len(voxelCoordinates)-n_minibatches*size_minibatches)  
    
    #########################################################################
    print('SEGMENTATION : Extracting {} image patches..'.format(total_number_of_patches))

    patches = extractImagePatch_parallelization(segmentChannels, voxelCoordinates, shape, segmentation_dpatch, 
                                                intensity_normalization_method, fullSegmentationPhase=True)    


    print('SEGMENTATION : Finished sampling data.')
#    if debug_coords:
#        patches = np.ones(patches.shape)
#        spatial_coordinates = np.zeros(spatial_coordinates.shape)
    
    INPUT_DATA = []  
    
    # NEED TO ADAPT FOR BREAST MASK MODEL: Inputs: Context (no resizing, 13,75,75) and spatial coordinates.
    if using_breastMaskModel:
        INPUT_DATA.append(patches[:,:,:,:,0].reshape(patches[:,:,:,:,0].shape + (1,)))  
        INPUT_DATA.append(spatial_coordinates)    
  
    elif using_Unet:  # This means the model is the U-Net
        if using_unet_breastMask:
            patches = patches[:,:,:,:,0]
            patches = patches.reshape(patches.shape + (1,))
        INPUT_DATA.append(patches)
#        if len(TPM_patches) > 0:
#            INPUT_DATA.append(TPM_patches[:,:,:,:].reshape(TPM_patches[:,:,:,:].shape + (1,)))   
        if len(spatial_coordinates) > 0:
            INPUT_DATA.append(spatial_coordinates)            
            
        
    else:
        # Context
        context = np.array(patches[:,:,:,:,0],'float')
        context = resize(image=context, order=1, 
                             output_shape=(context.shape[0],context.shape[1],context.shape[2]/3,context.shape[3]/3), 
                             anti_aliasing=True, preserve_range=True )
        INPUT_DATA.append(context.reshape(context.shape + (1,)))        
        
        for jj in range(patches.shape[-1]):
            INPUT_DATA.append(patches[:,:,model_crop/2:-model_crop/2,model_crop/2:-model_crop/2,jj].reshape(patches[:,:,model_crop/2:-model_crop/2,model_crop/2:-model_crop/2,jj].shape + (1,)))  
        if len(TPM_patches) > 0:
            INPUT_DATA.append(TPM_patches[:,:,:,:].reshape(TPM_patches[:,:,:,:].shape + (1,)))   
        if len(spatial_coordinates) > 0:
            INPUT_DATA.append(spatial_coordinates)    
    
    print("SEGMENTATION : Finished preprocessing data for segmentation.")
    #########################################################################
      
    prediction = model.predict(INPUT_DATA, verbose=1, batch_size=size_minibatches)

    ##########  Output probabilities ############

    indexes = []        
    class_pred = prediction[:,:,:,:,1]
    indexes.extend(class_pred)     
           
    img_probs = np.ones(shape, dtype=np.float32)  # same size as input head, start index for segmentation start at 26,26,26, rest filled with zeros....
    i = 0
    for x,y,z in voxelCoordinates:
        patch_shape = img_probs[x-output_dpatch[0]/2:min(x+(output_dpatch[0]/2+output_dpatch[0]%2), shape[0]),
                           y-output_dpatch[1]/2:min(y+(output_dpatch[1]/2+output_dpatch[1]%2), shape[1]),
                           z-output_dpatch[2]/2:min(z+(output_dpatch[2]/2+output_dpatch[2]%2), shape[2])].shape
        #print(np.array(indexes[i])[0:patch_shape[0], 0:patch_shape[1],0:patch_shape[2]])
        img_probs[x-output_dpatch[0]/2:min(x+(output_dpatch[0]/2+output_dpatch[0]%2), shape[0]),
             y-output_dpatch[1]/2:min(y+(output_dpatch[1]/2+output_dpatch[1]%2), shape[1]),
             z-output_dpatch[2]/2:min(z+(output_dpatch[2]/2+output_dpatch[2]%2), shape[2])] = np.array(indexes[i])[0:patch_shape[0], 
                                                                                                                   0:patch_shape[1],
                                                                                                                   0:patch_shape[2]]
        i = i+1

    img_probs = nib.Nifti1Image(img_probs, affine)
    return img_probs


####################################################################################################################################################################
####################################################################################################################################################################
#%%    Load Model, get intermediate layer, get features
my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
                                 'dice_coef_multilabel_bin0':dice_coef_multilabel_bin0,
                                 'dice_coef_multilabel_bin1':dice_coef_multilabel_bin1}

model = load_model(PATH_TO_MODEL, custom_objects = my_custom_objects)

#%%     Prediction 

#ALL_SEGMENT_CHANNELS = [segmentChannels2,segmentChannels3,segmentChannels4,segmentChannels5]
#ALL_SEGMENT_LABELS = [segmentLabels2,segmentLabels3,segmentLabels4,segmentLabels5]
#
#SEGMENTED_SLICE = 49
#for i in range(len(ALL_SEGMENT_CHANNELS)):
#    
#    segmentChannels = ALL_SEGMENT_CHANNELS[i]
#    segmentLabels = ALL_SEGMENT_LABELS[i]

subID = segmentChannels[0].split('/')[-2] + '_' + segmentChannels[0].split('/')[-1].split('.nii')[0]

img = fullSegmentation_quick(penalty_MATRIX, resolution, TPM_channel, model, 
                     segmentChannels, segmentLabels, output_classes, segmentation_dpatch, size_minibatches, use_coordinates, 
                     intensity_normalization_method, model_patch_reduction, model_crop, using_Unet=True)
 
#%%     Visualization of result

seg = img.get_data()
GT_Segmentation = nib.load(segmentLabels).get_data()
T1post_img = nib.load(segmentChannels[0]).get_data()
slope1_img = nib.load(segmentChannels[1]).get_data()
slope2_img = nib.load(segmentChannels[2]).get_data()

SEGMENTED_SLICE = np.mean(np.argwhere(GT_Segmentation > 0),0).astype(int)[0]

plt.figure(figsize=(10,16))
plt.subplot(5,1,1)
plt.imshow(GT_Segmentation[SEGMENTED_SLICE])
plt.subplot(5,1,2)
plt.imshow(seg[SEGMENTED_SLICE]   > 0.5)
plt.subplot(5,1,3)
plt.imshow(T1post_img[SEGMENTED_SLICE])
plt.subplot(5,1,4)
plt.imshow(slope1_img[SEGMENTED_SLICE])
plt.subplot(5,1,5)
plt.imshow(slope2_img[SEGMENTED_SLICE])
plt.tight_layout()
plt.savefig('/media/deeperthought/AXIAL/Segmentations/Test_Segmentation_{}.png'.format(i), dpi=200)
plt.close()

#%%          Get Dice score 
seg_out = resize(seg, T1post_img.shape, anti_aliasing=True, preserve_range=True) 
GT_Segmentation[GT_Segmentation>0] = 1
seg_out[seg_out>0.5] = 1
seg_out[seg_out<=0.5] = 0
print('Whole image dice: {}'.format(generalized_dice_completeImages(GT_Segmentation, seg_out)))
print('Selected slice dice: {}'.format(generalized_dice_completeImages(GT_Segmentation[SEGMENTED_SLICE], seg_out[SEGMENTED_SLICE])))

#%%          Save prediction

nifti = nib.load(segmentChannels[1])
out = nib.Nifti1Image(seg_out, nifti.affine)

nib.save(out, '/media/deeperthought/AXIAL/Segmentations/RIA_19-093_000_00038_20161114_seg_rotated_MODEL_PREDICTION_{}'.format(i))


