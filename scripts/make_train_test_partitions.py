# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 16:51:25 2018

@author: hirsch
"""

directory = '/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/CV_folds/'

T1post_PATH = directory + 'T1post.txt'
T1pre_PATH  = directory + 'T1pre.txt'
T2_PATH     = directory + 'T2.txt'
Labels_PATH = directory + 'LABELS.txt'


n_validation = 2000

SPLIT = 6  # In how many parts to split data

def partition_save(mris, name, x, n_validation, directory):

  test = [mris[i] for i in parts[x]]
  train =  [mris[i] for i in indexes if i not in parts[x]]  
  validation = train[-n_validation:]  
  train = train[:-n_validation]
  
  f = open(directory + '/test_{}.txt'.format(name),'a')
  for item in test:
      f.write(item)
  f.close()
  f = open(directory + '/train_{}.txt'.format(name),'a')
  for item in train:
     f.write(item)
  f.close()
  f = open(directory + '/validation_{}.txt'.format(name),'a')
  for item in validation:
     f.write(item)
  f.close()

######################################################################

T1post = open(T1post_PATH).readlines()
T1pre = open(T1pre_PATH).readlines()
T2 = open(T2_PATH).readlines()
labels = open(Labels_PATH).readlines()

indexes =range(len(T1post))

from random import shuffle
shuffle(indexes)
len(indexes)
import numpy as np
parts = np.array_split(indexes, SPLIT)
len(parts[0])

x = 0

partition_save(T1post,'t1post', x, n_validation, directory)
partition_save(T1pre,  't1pre', x, n_validation, directory)
partition_save(T2,  't2', x, n_validation, directory)
partition_save(labels, 'label', x, n_validation, directory)


    
