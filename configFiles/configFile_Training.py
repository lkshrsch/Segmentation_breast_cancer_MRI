#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import os
wd = os.getcwd()

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"
tf.keras.backend.set_session(tf.Session(config=config))

###################   parameters // replace with config files ########################

dataset = 'MyData'  # Custom name for session

############################## Load dataset #############################
 
#TPM_channel = ''
    
MRI_PATH = '/my/data/path/'

TPM_channel = ''

trainChannels = ['/CV_folds/TRAINING_IMAGES_T1.txt',
		 '/CV_folds/TRAINING_IMAGES_T2.txt']

trainLabels   = '/CV_folds/TRAINING_IMAGES_LABELS.txt'
    
testChannels  = ['/CV_folds/VALIDATION_IMAGES_T1.txt',
		 '/CV_folds/VALIDATION_IMAGES_T2.txt']

testLabels = 'VALIDATION_IMAGES_LABELS.txt'

validationChannels = testChannels
validationLabels = testLabels

output_classes = 2
test_subjects = 50
    
USE_PREPARED_DATA = False 
# Only use if have data pre-split into patches that can be fed directly into the network. Else will sample patches from full size MRIs.
PREPARED_DATA_FOLDER = ''


#################### MODEL PARAMETERS #####################

model = 'UNet_3D_v4' # Will import this model name, from main function in lib.py
dpatch= [19,43,43]
segmentation_dpatch = [27, 75, 75]
model_patch_reduction = [18,38,38]
model_crop = 0 
n_base_filters = 128

USE_UNCERTAINTY_LABEL_LOSS = False  
using_unet = True
using_unet_breastMask = False
resolution = 'high'

L2 = 0
# Loss functions: 'Dice', 'wDice', 'Multinomial'
loss_function = 'Dice'

load_model = False
path_to_model = ''
if load_model:
	session =  path_to_model.split('/')[-3]

num_channels = len(trainChannels)
dropout = [0,0]  # dropout for last two fully connected layers
learning_rate = 1e-05 
optimizer_decay = 0

##################### TRAIN PARAMETERS #####################
num_iter = 100
epochs = 100

#---- Dataset/Model related parameters ----
samplingMethod_train = 1
samplingMethod_val = 1
use_coordinates = False

merge_breastMask_model = False
path_to_breastMask_model = ''
Context_parameters_trainable = False

sample_intensity_based = True 
percentile_voxel_intensity_sample_benigns = 90

balanced_sample_subjects = True 		# SET TO FALSE WHEN TRAINING DATA HAS NO MALIGNANT/BENGING LABEL (breast mask model)
proportion_malignants_to_sample_train = 0.25
proportion_malignants_to_sample_val = 0.5
#------------------------------------------
n_subjects = 100
n_patches = 1000
size_minibatches = 16

data_augmentation = True 
proportion_to_flip = 0.5


# Intensity homogeneization method [1,2,3]
# 1 = normalization_by_histogram_matching_with_LUT_cluster
# 2 = percentile_normalization
# 3 = z_scoring_normalization

intensity_normalization_method = 2  

verbose = False 
quickmode = False        # Train without validation. Full segmentation often but only report dice score (whole)
n_subjects_val = 200  
n_patches_val = 1000 
size_minibatches_val = 16

INDEX_START_MALIGNANTS = 0
INDEX_START_BENIGNS = 0

####################### TEST PARAMETERS ####################
output_probability = True   # not thresholded network output for full scan segmentation
quick_segmentation = True
OUTPUT_PATH = ''
n_full_segmentations = 120
full_segmentation_patches = True
size_test_minibatches = 1
list_subjects_fullSegmentation = np.arange(0,120,1) # Leave empty if random   
epochs_for_fullSegmentation = np.arange(0,epochs+1,1) 
saveSegmentation = True 
proportion_malignants_fullSegmentation = 0.5

threshold_EARLY_STOP = 0

penalty_MATRIX = np.array([[ 1,  0],
			   [ 0,  1]], dtype='float32')


comments = ''

