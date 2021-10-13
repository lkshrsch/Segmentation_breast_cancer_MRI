#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:34:06 2018

@author: lukas
"""

import os
wd = os.getcwd()

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"
tf.keras.backend.set_session(tf.Session(config=config))

###################   parameters // replace with config files ########################

dataset = 'MyData'

############################## Load dataset #############################
 
MRI_PATH = '/my/data/path/'

TPM_channel = ''

segmentChannels = ['/CV_folds/TEST_IMAGES_T1.txt',
		   '/CV_folds/TEST_IMAGES_T2.txt']

segmentLabels = '/CV_folds/TEST_IMAGES_LABELS'  # leave empty if doing prediction on unlabeled data

output_classes = 2
    
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS

model = 'UNet_3D_v4' 
dpatch= [19,75,75]
segmentation_dpatch = [27,99,99]   # Can increase or reduce any dimensions by 8
model_patch_reduction = [18,38,38]
model_crop = 0 

use_uncertainty_label_loss = False
using_unet = True
using_unet_breastMask = False
resolution = 'high'

intensity_normalization_method = 2

path_to_model = '/path/to/trained/model.h5'
session =  path_to_model.split('/')[-3]

########################################### TEST PARAMETERS
use_coordinates = False
quick_segmentation = True
output_probability = True 
dice_compare = False
save_as_nifti = True  
OUTPUT_PATH = '/path/to/output/segmentations/'
percentile_normalization = True
full_segmentation_patches = True
test_subjects = 250
n_full_segmentations = 250
list_subjects_fullSegmentation = range(250)
size_test_minibatches = 8
saveSegmentation = True

import numpy as np
penalty_MATRIX = np.array([[ 1,  0],
			   [ 0,  1]], dtype='float32')

comments = ''

