#!/usr/bin/env python3
from numpy import float64
from tensorflow.contrib.losses.python.metric_learning.metric_loss_ops import npairs_loss

__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "andrey@elphel.com"


from PIL import Image

import os
import sys
import glob

import numpy as np
import itertools

import time

import matplotlib.pyplot as plt

import shutil
import sys

TIME_START = time.time()
TIME_LAST  = TIME_START
DEBUG_LEVEL= 1
DISP_BATCH_BINS =   20 # Number of batch disparity bins
STR_BATCH_BINS =    10 # Number of batch strength bins
FILES_PER_SCENE =    5 # number of random offset files for the scene to select from (0 - use all available)
#MIN_BATCH_CHOICES = 10 # minimal number of tiles in a file for each bin to select from 
#MAX_BATCH_FILES =   10 #maximal number of files to use in a batch
#MAX_EPOCH =        500
LR =               1e-3 # learning rate
LR100 =            LR    # 1e-4
LR200 =            LR100 # 3e-5
LR400 =            LR200 # 1e-5
LR600 =            LR400 # 3e-6
USE_CONFIDENCE =     False
ABSOLUTE_DISPARITY = False # True # False # True # False
DEBUG_PLT_LOSS =     True
TILE_LAYERS =        4
FILE_TILE_SIDE =     9
TILE_SIDE =          9 # 7
TILE_SIZE =         TILE_SIDE* TILE_SIDE # == 81
FEATURES_PER_TILE =  TILE_LAYERS * TILE_SIZE# == 324
EPOCHS_TO_RUN =     3000#0 #0
EPOCHS_FULL_TEST =   5 # 10 # 25# repeat full image test after this number of epochs
#EPOCHS_SAME_FILE =   20
RUN_TOT_AVG =       100 # last batches to average. Epoch is 307 training  batches  
#BATCH_SIZE =       2*1080//9 # == 120 Each batch of tiles has balanced D/S tiles, shuffled batches but not inside batches
BATCH_SIZE =       2*1000//25 # == 80 Each batch of tiles has balanced D/S tiles, shuffled batches but not inside batches
SHUFFLE_EPOCH =    True
NET_ARCH1 =        0 #3 # 0 #  0 # 0 # 0 # 8 # 0 # 7 # 2 #0 # 6 #0 # 4 # 3 # overwrite with argv?
NET_ARCH2 =        0 # 0 # 3 #  0 # 3 # 0 # 3 # 0 # 2 #0 # 6 # 0 # 3 # overwrite with argv?
SYM8_SUB =         False # True # False # True # False # enforce inputs from 2d correlation have symmetrical ones (groups of 8)
ONLY_TILE =        None # 4 # None # 0 # 4# None # (remove all but center tile data), put None here for normal operation)
ZIP_LHVAR =        True # combine _lvar and _hvar as odd/even elements 

#DEBUG_PACK_TILES = True
SUFFIX=str(NET_ARCH1)+'-'+str(NET_ARCH2)+ (["R","A"][ABSOLUTE_DISPARITY]) +(["NS","S8"][SYM8_SUB])
# CLUSTER_RADIUS should match input data
CLUSTER_RADIUS =     2 # 1 # 1 - 3x3, 2 - 5x5 tiles
SHUFFLE_FILES  =     True

NN_LAYOUTS = {0:[0,   0,   0,   32,  20,  16],
              1:[0,   0,   0,  256, 128,  64],
              2:[0, 128,  32,   32,  32,  16],
              3:[0,   0,  40,   32,  20,  16],
              4:[0,   0,   0,    0,  16,  16],
              5:[0,   0,  64,   32,  32,  16],
              6:[0,   0,  32,   16,  16,  16],
              7:[0,   0,  64,   16,  16,  16],
              8:[0,   0,   0,   64,  20,  16],
              }
NN_LAYOUT1 = NN_LAYOUTS[NET_ARCH1]
NN_LAYOUT2 = NN_LAYOUTS[NET_ARCH2]

#http://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[38;5;214m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    BOLDWHITE = '\033[1;37m'
    UNDERLINE = '\033[4m'
def print_time(txt="",end="\n"):
    global TIME_LAST
    t = time.time()
    if txt:
        txt +=" "
    print(("%s"+bcolors.BOLDWHITE+"at %.4fs (+%.4fs)"+bcolors.ENDC)%(txt,t-TIME_START,t-TIME_LAST), end = end, flush=True)
    TIME_LAST = t
#reading to memory (testing)
def readTFRewcordsEpoch(train_filename):
#    filenames = [train_filename]
#    dataset = tf.data.TFRecorDataset(filenames)
    if not  '.tfrecords' in train_filename:
        train_filename += '.tfrecords'
    npy_dir_name = "npy"
    dirname = os.path.dirname(train_filename) 
    npy_dir = os.path.join(dirname, npy_dir_name)
    filebasename, file_extension = os.path.splitext(train_filename)
    filebasename = os.path.basename(filebasename)
    file_corr2d =           os.path.join(npy_dir,filebasename + '_corr2d.npy')
    file_target_disparity = os.path.join(npy_dir,filebasename + '_target_disparity.npy')
    file_gt_ds =            os.path.join(npy_dir,filebasename + '_gt_ds.npy')
    if (os.path.exists(file_corr2d) and
        os.path.exists(file_target_disparity) and
        os.path.exists(file_gt_ds)):
        corr2d=            np.load (file_corr2d)
        target_disparity = np.load(file_target_disparity)
        gt_ds =            np.load(file_gt_ds)
        pass
    else:     
        record_iterator = tf.python_io.tf_record_iterator(path=train_filename)
        corr2d_list=[]
        target_disparity_list=[]
        gt_ds_list = []
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            corr2d_list.append           (np.array(example.features.feature['corr2d'].float_list.value, dtype=np.float32))
            target_disparity_list.append (np.array(example.features.feature['target_disparity'].float_list.value, dtype=np.float32))
            gt_ds_list.append            (np.array(example.features.feature['gt_ds'].float_list.value, dtype= np.float32))
            pass
        corr2d=            np.array(corr2d_list)
        target_disparity = np.array(target_disparity_list)
        gt_ds =            np.array(gt_ds_list)
        try:
            os.makedirs(os.path.dirname(file_corr2d))
        except:
            pass     

        np.save(file_corr2d,           corr2d)
        np.save(file_target_disparity, target_disparity)
        np.save(file_gt_ds,            gt_ds)
        
    return corr2d, target_disparity, gt_ds   

#from http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'corr2d':           tf.FixedLenFeature([FEATURES_PER_TILE],tf.float32), #string),
        'target_disparity': tf.FixedLenFeature([1],   tf.float32), #.string),
        'gt_ds':            tf.FixedLenFeature([2],  tf.float32)  #.string)
        })
    corr2d =           features['corr2d'] # tf.decode_raw(features['corr2d'], tf.float32)
    target_disparity = features['target_disparity'] # tf.decode_raw(features['target_disparity'], tf.float32)
    gt_ds =            tf.cast(features['gt_ds'], tf.float32) # tf.decode_raw(features['gt_ds'], tf.float32)
    in_features = tf.concat([corr2d,target_disparity],0)
    # still some nan-s in correlation data?
#    in_features_clean = tf.where(tf.is_nan(in_features), tf.zeros_like(in_features), in_features)     
#    corr2d_out, target_disparity_out, gt_ds_out = tf.train.shuffle_batch( [in_features_clean, target_disparity, gt_ds],
    corr2d_out, target_disparity_out, gt_ds_out = tf.train.shuffle_batch( [in_features, target_disparity, gt_ds],
                                                 batch_size=1000, # 2,
                                                 capacity=30,
                                                 num_threads=2,
                                                 min_after_dequeue=10)
    return corr2d_out, target_disparity_out, gt_ds_out
#http://adventuresinmachinelearning.com/introduction-tensorflow-queuing/
def add_margins(npa,radius, val = np.nan):
    npa_ext = np.empty((npa.shape[0]+2*radius, npa.shape[1]+2*radius, npa.shape[2]), dtype = npa.dtype)
    npa_ext[radius:radius + npa.shape[0],radius:radius + npa.shape[1]] = npa
    npa_ext[0:radius,:,:] = val  
    npa_ext[radius + npa.shape[0]:,:,:] = val  
    npa_ext[:,0:radius,:] = val  
    npa_ext[:, radius + npa.shape[1]:,:] = val
    return npa_ext   

def add_neibs(npa_ext,radius):
    height = npa_ext.shape[0]-2*radius
    width =  npa_ext.shape[1]-2*radius
    side = 2 * radius + 1
    size = side * side
    npa_neib = np.empty((height,  width, side, side, npa_ext.shape[2]), dtype = npa_ext.dtype)
    for dy in range (side):
        for dx in range (side):
            npa_neib[:,:,dy, dx,:]= npa_ext[dy:dy+height, dx:dx+width]
    return npa_neib.reshape(height, width, -1)    
    
def extend_img_to_clusters(datasets_img,radius):
    side = 2 * radius + 1
    size = side * side
    width =  324
    if len(datasets_img) ==0:
        return
    num_tiles = datasets_img[0]['corr2d'].shape[0]
    height = num_tiles // width 
    for rec in datasets_img:
        rec['corr2d'] =           add_neibs(add_margins(rec['corr2d'].reshape((height,width,-1)), radius, np.nan), radius).reshape((num_tiles,-1)) 
        rec['target_disparity'] = add_neibs(add_margins(rec['target_disparity'].reshape((height,width,-1)), radius, np.nan), radius).reshape((num_tiles,-1)) 
        rec['gt_ds'] =            add_neibs(add_margins(rec['gt_ds'].reshape((height,width,-1)), radius, np.nan), radius).reshape((num_tiles,-1))
        pass



def reformat_to_clusters(datasets_data):
    cluster_size = (2 * CLUSTER_RADIUS + 1) * (2 * CLUSTER_RADIUS + 1)
# Reformat input data
    for rec in datasets_data:
        rec['corr2d'] =           rec['corr2d'].reshape(          (rec['corr2d'].shape[0]//cluster_size,           rec['corr2d'].shape[1] * cluster_size)) 
        rec['target_disparity'] = rec['target_disparity'].reshape((rec['target_disparity'].shape[0]//cluster_size, rec['target_disparity'].shape[1] * cluster_size)) 
        rec['gt_ds'] =            rec['gt_ds'].reshape(           (rec['gt_ds'].shape[0]//cluster_size,            rec['gt_ds'].shape[1] * cluster_size))

def replace_nan(datasets_data):
    cluster_size = (2 * CLUSTER_RADIUS + 1) * (2 * CLUSTER_RADIUS + 1)
# Reformat input data
    for rec in datasets_data:
        np.nan_to_num(rec['corr2d'],           copy = False) 
        np.nan_to_num(rec['target_disparity'], copy = False) 
        np.nan_to_num(rec['gt_ds'],            copy = False)

def permute_to_swaps(perm):
    pairs = []
    for i in range(len(perm)):
        w = np.where(perm == i)[0][0]
        if w != i:
            pairs.append([i,w])
            perm[w] = perm[i]
            perm[i] = i
    return pairs        
        
def shuffle_in_place(datasets_data, indx, period):
    swaps = permute_to_swaps(np.random.permutation(len(datasets_data)))
    num_entries = datasets_data[0]['corr2d'].shape[0] // period
    for swp in swaps:
        ds0 = datasets_data[swp[0]]
        ds1 = datasets_data[swp[1]]
        tmp = ds0['corr2d'][indx::period].copy() 
        ds0['corr2d'][indx::period] = ds1['corr2d'][indx::period]
        ds1['corr2d'][indx::period] = tmp

        tmp = ds0['target_disparity'][indx::period].copy() 
        ds0['target_disparity'][indx::period] = ds1['target_disparity'][indx::period]
        ds1['target_disparity'][indx::period] = tmp

        tmp = ds0['gt_ds'][indx::period].copy() 
        ds0['gt_ds'][indx::period] = ds1['gt_ds'][indx::period]
        ds1['gt_ds'][indx::period] = tmp
    
    
    
def zip_lvar_hvar(datasets_lvar_data, datasets_hvar_data, del_src = True, shuffle = False):
#    cluster_size = (2 * CLUSTER_RADIUS + 1) * (2 * CLUSTER_RADIUS + 1)
# Reformat input data

    datasets_data = []
    permut_l = np.random.permutation(len(datasets_lvar_data))    
    permut_h = np.random.permutation(len(datasets_hvar_data))
    """
    TODO: add shuffling batches in each record
    """    
#    for rec1, rec2 in zip(datasets_lvar_data, datasets_hvar_data):
    for nrec in range(len(datasets_lvar_data)):
        if shuffle:
            rec1 = datasets_lvar_data[permut_l[nrec]]
            rec2 = datasets_hvar_data[permut_h[nrec]]
        else:
            rec1 = datasets_lvar_data[nrec]
            rec2 = datasets_hvar_data[nrec]
            
#        print ("\n",nrec, permut_l[nrec], permut_h[nrec])
#        print (rec1['corr2d'].shape,rec2['corr2d'].shape)
        rec = {'corr2d':           np.empty((rec1['corr2d'].shape[0]*2,rec1['corr2d'].shape[1]),dtype=np.float32),
               'target_disparity': np.empty((rec1['target_disparity'].shape[0]*2,rec1['target_disparity'].shape[1]),dtype=np.float32),
               'gt_ds':            np.empty((rec1['gt_ds'].shape[0]*2,rec1['gt_ds'].shape[1]),dtype=np.float32)}

        rec['corr2d'][0::2] =           rec1['corr2d'] 
        rec['corr2d'][1::2] =           rec2['corr2d'] 
        
        rec['target_disparity'][0::2] = rec1['target_disparity'] 
        rec['target_disparity'][1::2] = rec2['target_disparity'] 

        rec['gt_ds'][0::2] =            rec1['gt_ds'] 
        rec['gt_ds'][1::2] =            rec2['gt_ds']
        if del_src:
            datasets_lvar_data[nrec] = None
            datasets_hvar_data[nrec] = None
        datasets_data.append(rec)
    return datasets_data    
         

# list of dictionaries  
def reduce_tile_size(datasets_data, num_tile_layers, reduced_tile_side):
    if (not datasets_data is None) and (len (datasets_data) > 0): 
        tsz = (datasets_data[0]['corr2d'].shape[1])// num_tile_layers # 81 # list index out of range
        tss = int(np.sqrt(tsz)+0.5)
        offs = (tss - reduced_tile_side) // 2
        for rec in datasets_data:
            rec['corr2d'] =  (rec['corr2d'].reshape((-1, num_tile_layers,  tss, tss))
                                 [..., offs:offs+reduced_tile_side, offs:offs+reduced_tile_side].
                                 reshape(-1,num_tile_layers*reduced_tile_side*reduced_tile_side))

def eval_results(rslt_path, absolute,
                 min_disp =       -0.1, #minimal GT disparity
                 max_disp =       20.0, # maximal GT disparity
                 max_ofst_target = 1.0,
                 max_ofst_result = 1.0,
                 str_pow =         2.0,
                 radius =          0):
#    for min_disparity, max_disparity, max_offset_target, max_offset_result, strength_pow in [
    variants = [[         -0.1,         5.0,              0.5,            0.5,          1.0],           
                [         -0.1,         5.0,              0.5,            0.5,          2.0],
                [         -0.1,         5.0,              0.2,            0.2,          1.0],
                [         -0.1,         5.0,              0.2,            0.2,          2.0],
                [         -0.1,        20.0,              0.5,            0.5,          1.0],           
                [         -0.1,        20.0,              0.5,            0.5,          2.0],
                [         -0.1,        20.0,              0.2,            0.2,          1.0],
                [         -0.1,        20.0,              0.2,            0.2,          2.0],
                [         -0.1,        20.0,              1.0,            1.0,          1.0],
                [min_disp, max_disp, max_ofst_target, max_ofst_result, str_pow]]
    

    rslt = np.load(result_file)
    not_nan  =  ~np.isnan(rslt[...,0])
    not_nan &=  ~np.isnan(rslt[...,1])
    not_nan &=  ~np.isnan(rslt[...,2])
    not_nan &=  ~np.isnan(rslt[...,3])
    not_nan_ext = np.zeros((rslt.shape[0] + 2*radius,rslt.shape[1] + 2 * radius),dtype=np.bool) 
    not_nan_ext[radius:-radius,radius:-radius] = not_nan
    for dy in range(2*radius+1):
        for dx in range(2*radius+1):
            not_nan_ext[dy:dy+not_nan.shape[0], dx:dx+not_nan.shape[1]] &= not_nan
    not_nan = not_nan_ext[radius:-radius,radius:-radius]         
        
    if  not absolute:
        rslt[...,0] +=  rslt[...,1]
    nn_disparity =     np.nan_to_num(rslt[...,0], copy = False)
    target_disparity = np.nan_to_num(rslt[...,1], copy = False)
    gt_disparity =     np.nan_to_num(rslt[...,2], copy = False)
    gt_strength =      np.nan_to_num(rslt[...,3], copy = False)
    rslt = []
    for min_disparity, max_disparity, max_offset_target, max_offset_result, strength_pow in variants:
        good_tiles = not_nan.copy();
        good_tiles &= (gt_disparity >= min_disparity)
        good_tiles &= (gt_disparity <= max_disparity)
        good_tiles &= (target_disparity != gt_disparity)
        good_tiles &= (np.abs(target_disparity - gt_disparity) <= max_offset_target)
        good_tiles &= (np.abs(target_disparity - nn_disparity) <= max_offset_result)
        gt_w =  gt_strength * good_tiles
        gt_w = np.power(gt_w,strength_pow)
        sw = gt_w.sum() 
        diff0 = target_disparity - gt_disparity
        diff1 = nn_disparity -     gt_disparity
        diff0_2w = gt_w*diff0*diff0
        diff1_2w = gt_w*diff1*diff1
        rms0 = np.sqrt(diff0_2w.sum()/sw) 
        rms1 = np.sqrt(diff1_2w.sum()/sw)
        print ("%7.3f<disp<%7.3f, offs_tgt<%5.2f, offs_rslt<%5.2f pwr=%05.3f, rms0=%7.4f, rms1=%7.4f (gain=%7.4f) num good tiles = %5d"%(
            min_disparity, max_disparity, max_offset_target,  max_offset_result, strength_pow, rms0, rms1, rms0/rms1, good_tiles.sum() ))
        rslt.append([rms0,rms1])
    return rslt 
    
                         

"""
Start of the main code
"""
"""
try:
    train_filenameTFR =  sys.argv[1]
except IndexError:
    train_filenameTFR = "/mnt/dde6f983-d149-435e-b4a2-88749245cc6c/home/eyesis/x3d_data/data_sets/tf_data/train_00.tfrecords"
try:
    test_filenameTFR =  sys.argv[2]
except IndexError:
    test_filenameTFR = "/mnt/dde6f983-d149-435e-b4a2-88749245cc6c/home/eyesis/x3d_data/data_sets/tf_data/test.tfrecords"
#FILES_PER_SCENE
train_filenameTFR1 = "/mnt/dde6f983-d149-435e-b4a2-88749245cc6c/home/eyesis/x3d_data/data_sets/tf_data/train_01.tfrecords"
"""
data_dir =  "/home/eyesis/x3d_data/data_sets/tf_data_5x5_main_1" # "/home/eyesis/x3d_data/data_sets/tf_data_5x5_dbg" #"/home/eyesis/x3d_data/data_sets/tf_data_5x5_dbg" 
img_dir =   "/home/eyesis/x3d_data/data_sets/tf_data_5x5_main_1" # "/home/eyesis/x3d_data/data_sets/tf_data_5x5_dbg"

dir_train_lvar = data_dir #"/home/eyesis/x3d_data/data_sets/tf_data_5x5_main_1" # data_dir # "/home/eyesis/x3d_data/data_sets/tf_data_5x5_main"
dir_train_hvar = data_dir # "/home/eyesis/x3d_data/data_sets/tf_data_5x5_main"
dir_test_lvar =  data_dir # "/home/eyesis/x3d_data/data_sets/tf_data_5x5_main_2" # data_dir #"/home/eyesis/x3d_data/data_sets/tf_data_5x5_center/" # data_dir # "/home/eyesis/x3d_data/data_sets/tf_data_5x5_main"
dir_test_hvar =  "/home/eyesis/x3d_data/data_sets/tf_data_5x5_main_2" # data_dir #"/home/eyesis/x3d_data/data_sets/tf_data_5x5_center" #  data_dir # "/home/eyesis/x3d_data/data_sets/tf_data_5x5_main"
dir_img =        os.path.join(img_dir,"img") # "/home/eyesis/x3d_data/data_sets/tf_data_5x5_main/img"
dir_result =     os.path.join(data_dir,"result") # "/home/eyesis/x3d_data/data_sets/tf_data_5x5_main/result"

files_train_lvar = ["train000_R2_LE_1.5.tfrecords",
                    "train001_R2_LE_1.5.tfrecords",
                    "train002_R2_LE_1.5.tfrecords",
                    "train003_R2_LE_1.5.tfrecords",
                    "train004_R2_LE_1.5.tfrecords",
                    "train005_R2_LE_1.5.tfrecords",
                    "train006_R2_LE_1.5.tfrecords",
                    "train007_R2_LE_1.5.tfrecords",
                    ]
files_train_hvar = ["train000_R2_GT_1.5.tfrecords",
                    "train001_R2_GT_1.5.tfrecords",
                    "train002_R2_GT_1.5.tfrecords",
                    "train003_R2_GT_1.5.tfrecords",
                    "train004_R2_GT_1.5.tfrecords",
                    "train005_R2_GT_1.5.tfrecords",
                    "train006_R2_GT_1.5.tfrecords",
                    "train007_R2_GT_1.5.tfrecords",
]
"""
Try again - all hvar training, train than different hvar testing.
tensorboard --logdir="attic/nn_ds_neibs6_graph0-0RNS-bothtrain-test-hvar" --port=7069
Seems that even different (but used) hvar perfectly match each other, but training for both lvar and hvar never get
match for either of hvar/lvar, even those that were used for training
tensorboard --logdir="attic/nn_ds_neibs6_graph0-0RNS-19-tested_with_same_hvar_lvar_as_trained" --port=7070

try same with higher LR - will they eventually converge?

Compare with other (not used) train sets (use 7 of each instead of 8, 8-th as test)
"""
#just testing:
#files_train_lvar = files_train_hvar
files_test_lvar =  ["train004_R2_GT_1.5.tfrecords"]# ["train007_R2_LE_1.5.tfrecords"]# "testTEST_R2_LE_1.5.tfrecords"] # testTEST_R2_LE_1.5.tfrecords"]
files_test_hvar =  ["train004_R2_GT_1.5.tfrecords"]#"testTEST_R2_GT_1.5.tfrecords"] # "testTEST_R2_GT_1.5.tfrecords"]
#files_img =        ['1527257933_150165-v04'] # overlook
#files_img =        ['1527256858_150165-v01'] # state Street
files_img =        ['1527256816_150165-v02'] # staete Street
for i, path in enumerate(files_train_lvar):
    files_train_lvar[i]=os.path.join(dir_train_lvar, path)
    
for i, path in enumerate(files_train_hvar):
    files_train_hvar[i]=os.path.join(dir_train_hvar, path)
    
for i, path in enumerate(files_test_lvar):
    files_test_lvar[i]=os.path.join(dir_test_lvar, path)
    
for i, path in enumerate(files_test_hvar):
    files_test_hvar[i]=os.path.join(dir_test_hvar, path)

result_files=[]
for i, path in enumerate(files_img):
    files_img[i] =      os.path.join(dir_img,    path+'.tfrecords')
    result_files.append(os.path.join(dir_result, path+"_"+SUFFIX+'.npy'))


#file_test_hvar=  None
weight_hvar = 0.26
weight_lvar = 1.0 - weight_hvar 


import tensorflow as tf
import tensorflow.contrib.slim as slim


for result_file in result_files:
    try:
        eval_results(result_file, ABSOLUTE_DISPARITY,radius=CLUSTER_RADIUS)
    #exit(0)
    except:
        pass

datasets_img = []
for fpath in files_img:
    print_time("Importing test image data from "+fpath, end="")
    corr2d, target_disparity, gt_ds = readTFRewcordsEpoch(fpath)
    datasets_img.append(       {"corr2d":           corr2d,
                                "target_disparity": target_disparity,
                                "gt_ds":            gt_ds})
    print_time("  Done")
gtruth =   datasets_img[0]['gt_ds'].copy()
t_disp =   datasets_img[0]['target_disparity'].reshape([-1,1]).copy()
extend_img_to_clusters(datasets_img, radius = CLUSTER_RADIUS)
#reformat_to_clusters(datasets_img) already this format   
replace_nan(datasets_img)
pass
pass


datasets_train_lvar = []
for fpath in files_train_lvar:
    print_time("Importing train data (low variance) from "+fpath, end="")
    corr2d, target_disparity, gt_ds = readTFRewcordsEpoch(fpath)
    datasets_train_lvar.append({"corr2d":corr2d,
                                "target_disparity":target_disparity,
                                "gt_ds":gt_ds})
    print_time("  Done")
datasets_train_hvar = []
for fpath in files_train_hvar:
    print_time("Importing train data (high variance) from "+fpath, end="")
    corr2d, target_disparity, gt_ds = readTFRewcordsEpoch(fpath)
    datasets_train_hvar.append({"corr2d":corr2d,
                                "target_disparity":target_disparity,
                                "gt_ds":gt_ds})
    print_time("  Done")
    """
if (file_test_lvar):
    print_time("Importing test data (low variance) from "+fpath, end="")
    corr2d, target_disparity, gt_ds = readTFRewcordsEpoch(file_test_lvar)
    dataset_test_lvar = {"corr2d":corr2d,
                           "target_disparity":target_disparity,
                           "gt_ds":gt_ds}
    print_time("  Done")
if (file_test_hvar):
    print_time("Importing test data (high variance) from "+fpath, end="")
    corr2d, target_disparity, gt_ds = readTFRewcordsEpoch(file_test_hvar)
    dataset_test_hvar = {"corr2d":corr2d,
                           "target_disparity":target_disparity,
                           "gt_ds":gt_ds}
    print_time("  Done")
    """
datasets_test_lvar = []
for fpath in files_test_lvar:
    print_time("Importing test data (low variance) from "+fpath, end="")
    corr2d, target_disparity, gt_ds = readTFRewcordsEpoch(fpath)
    datasets_test_lvar.append({"corr2d":corr2d,
                                "target_disparity":target_disparity,
                                "gt_ds":gt_ds})
    print_time("  Done")

datasets_test_hvar = []
for fpath in files_test_hvar:
    print_time("Importing test data (high variance) from "+fpath, end="")
    corr2d, target_disparity, gt_ds = readTFRewcordsEpoch(fpath)
    datasets_test_hvar.append({"corr2d":corr2d,
                                "target_disparity":target_disparity,
                                "gt_ds":gt_ds})
    print_time("  Done")
    
    
    
    
#CLUSTER_RADIUS
cluster_size = (2 * CLUSTER_RADIUS + 1) * (2 * CLUSTER_RADIUS + 1)
center_tile_index = 2 * CLUSTER_RADIUS * (CLUSTER_RADIUS + 1)
# Reformat input data
if FILE_TILE_SIDE > TILE_SIDE:
    print_time("Reducing correlation tile size from %d to %d"%(FILE_TILE_SIDE, TILE_SIDE), end="")
    reduce_tile_size(datasets_train_lvar, TILE_LAYERS, TILE_SIDE)
    reduce_tile_size(datasets_train_hvar, TILE_LAYERS, TILE_SIDE)
    reduce_tile_size(datasets_test_lvar,  TILE_LAYERS, TILE_SIDE)
    reduce_tile_size(datasets_test_hvar,  TILE_LAYERS, TILE_SIDE)
    print_time("  Done")
pass
print_time("Reshaping train data (low variance)", end="")
reformat_to_clusters(datasets_train_lvar)
print_time("  Done")
print_time("Reshaping train data (high variance)", end="")
reformat_to_clusters(datasets_train_hvar)
print_time("  Done")
print_time("Reshaping test data (low variance)", end="")
reformat_to_clusters(datasets_test_lvar)
print_time("  Done")
print_time("Reshaping test data (high variance)", end="")
reformat_to_clusters(datasets_test_hvar)
print_time("  Done")
pass    

"""
datasets_train_lvar & datasets_train_hvar ( that will increase batch size and placeholders twice
test has to have even original, batches will not zip - just use two batches for one big one
"""




if ZIP_LHVAR:
    print_time("Zipping together datasets datasets_train_lvar and datasets_train_hvar", end="")
#    datasets_train = zip_lvar_hvar(datasets_train_lvar, datasets_train_hvar, del_src= not SHUFFLE_FILES, shuffle = SHUFFLE_FILES)
    datasets_train = zip_lvar_hvar(datasets_train_lvar, datasets_train_hvar) # no shuffle, delete src
    print_time("  Done")
else:
    #Alternate lvar/hvar
    datasets_train = []
    datasets_weights_train = []
    for indx in range(max(len(datasets_train_lvar),len(datasets_train_hvar))):
        if (indx < len(datasets_train_lvar)):
            datasets_train.append(datasets_train_lvar[indx])
            datasets_weights_train.append(weight_lvar)
        if (indx < len(datasets_train_hvar)):
            datasets_train.append(datasets_train_hvar[indx])
            datasets_weights_train.append(weight_hvar)

datasets_test = []
datasets_weights_test = []
for dataset_test_lvar in datasets_test_lvar:
    datasets_test.append(dataset_test_lvar)
    datasets_weights_test.append(weight_lvar)
for dataset_test_hvar in datasets_test_hvar:
    datasets_test.append(dataset_test_hvar)
    datasets_weights_test.append(weight_hvar)

    
corr2d_train_placeholder =           tf.placeholder(datasets_train[0]['corr2d'].dtype,           (None,FEATURES_PER_TILE * cluster_size)) # corr2d_train.shape)
target_disparity_train_placeholder = tf.placeholder(datasets_train[0]['target_disparity'].dtype, (None,1 *   cluster_size))  #target_disparity_train.shape)
gt_ds_train_placeholder =            tf.placeholder(datasets_train[0]['gt_ds'].dtype,            (None,2 *   cluster_size)) #gt_ds_train.shape)
#dataset_tt    TensorSliceDataset: <TensorSliceDataset shapes: {gt_ds: (50,), corr2d: (8100,), target_disparity: (25,)}, types: {gt_ds: tf.float32, corr2d: tf.float32, target_disparity: tf.float32}>    
dataset_tt = tf.data.Dataset.from_tensor_slices({
    "corr2d":corr2d_train_placeholder,
    "target_disparity": target_disparity_train_placeholder,
    "gt_ds": gt_ds_train_placeholder})


#dataset_train_size = len(corr2d_train)

#dataset_train_size = len(datasets_train_lvar[0]['corr2d'])
dataset_train_size = len(datasets_train[0]['corr2d'])
dataset_train_size //= BATCH_SIZE
dataset_test_size = len(datasets_test_lvar[0]['corr2d'])
dataset_test_size //= BATCH_SIZE
dataset_img_size = len(datasets_img[0]['corr2d'])
dataset_img_size //= BATCH_SIZE

#print_time("dataset_tt.output_types "+str(dataset_train.output_types)+", dataset_train.output_shapes "+str(dataset_train.output_shapes)+", number of elements="+str(dataset_train_size))
dataset_tt = dataset_tt.batch(BATCH_SIZE)
dataset_tt = dataset_tt.prefetch(BATCH_SIZE)
iterator_tt = dataset_tt.make_initializable_iterator()
next_element_tt = iterator_tt.get_next()
#print("dataset_tt.output_types "+str(dataset_tt.output_types)+", dataset_tt.output_shapes "+str(dataset_tt.output_shapes)+", number of elements="+str(dataset_train_size))
#BatchDataset: <BatchDataset shapes: {gt_ds: (?, 50), corr2d: (?, 8100), target_disparity: (?, 25)}, types: {gt_ds: tf.float32, corr2d: tf.float32, target_disparity: tf.float32}>
"""
next_element_tt    dict: {'gt_ds': <tf.Tensor 'IteratorGetNext:1' shape=(?, 50) dtype=float32>, 'corr2d': <tf.Tensor 'IteratorGetNext:0' shape=(?, 8100) dtype=float32>, 'target_disparity': <tf.Tensor 'IteratorGetNext:2' shape=(?, 25) dtype=float32>}    
    'corr2d' (140405473715624)    Tensor: Tensor("IteratorGetNext:0", shape=(?, 8100), dtype=float32)    
    'gt_ds' (140405473715680)    Tensor: Tensor("IteratorGetNext:1", shape=(?, 50), dtype=float32)    
    'target_disparity' (140405501995888)    Tensor: Tensor("IteratorGetNext:2", shape=(?, 25), dtype=float32)    
"""

#https://www.tensorflow.org/versions/r1.5/programmers_guide/datasets
result_dir = './attic/result_neibs_'+     SUFFIX+'/'
checkpoint_dir = './attic/result_neibs_'+ SUFFIX+'/'
save_freq = 500
def lrelu(x):
    return tf.maximum(x*0.2,x)
#    return tf.nn.relu(x)

def sym_inputs8(inp):
    """
    get input vector [?:4*9*9+1] (last being target_disparity) and reorder for horizontal flip,
    vertical flip and transpose (8 variants, mode + 1 - hor, +2 - vert, +4 - transpose)
    return same lengh, reordered
    """
    with tf.name_scope("sym_inputs8"):
        td =           inp[:,-1:] # tf.reshape(inp,[-1], name = "td")[-1]
        inp_corr =     tf.reshape(inp[:,:-1],[-1,4,TILE_SIDE,TILE_SIDE], name = "inp_corr")
        inp_corr_h =   tf.stack([-inp_corr  [:,0,:,-1::-1], inp_corr  [:,1,:,-1::-1], -inp_corr  [:,3,:,-1::-1], -inp_corr  [:,2,:,-1::-1]], axis=1, name = "inp_corr_h")
        inp_corr_v =   tf.stack([ inp_corr  [:,0,-1::-1,:],-inp_corr  [:,1,-1::-1,:],  inp_corr  [:,3,-1::-1,:],  inp_corr  [:,2,-1::-1,:]], axis=1, name = "inp_corr_v")
        inp_corr_hv =  tf.stack([ inp_corr_h[:,0,-1::-1,:],-inp_corr_h[:,1,-1::-1,:],  inp_corr_h[:,3,-1::-1,:],  inp_corr_h[:,2,-1::-1,:]], axis=1, name = "inp_corr_hv")
        inp_corr_t =   tf.stack([tf.transpose(inp_corr   [:,1], perm=[0,2,1]),
                                 tf.transpose(inp_corr   [:,0], perm=[0,2,1]),
                                 tf.transpose(inp_corr   [:,2], perm=[0,2,1]),
                                -tf.transpose(inp_corr   [:,3], perm=[0,2,1])], axis=1, name = "inp_corr_t")
        inp_corr_ht =  tf.stack([tf.transpose(inp_corr_h [:,1], perm=[0,2,1]),
                                 tf.transpose(inp_corr_h [:,0], perm=[0,2,1]),
                                 tf.transpose(inp_corr_h [:,2], perm=[0,2,1]),
                                -tf.transpose(inp_corr_h [:,3], perm=[0,2,1])], axis=1, name = "inp_corr_ht")
        inp_corr_vt =  tf.stack([tf.transpose(inp_corr_v [:,1], perm=[0,2,1]),
                                 tf.transpose(inp_corr_v [:,0], perm=[0,2,1]),
                                 tf.transpose(inp_corr_v [:,2], perm=[0,2,1]),
                                -tf.transpose(inp_corr_v [:,3], perm=[0,2,1])], axis=1, name = "inp_corr_vt")
        inp_corr_hvt = tf.stack([tf.transpose(inp_corr_hv[:,1], perm=[0,2,1]),
                                 tf.transpose(inp_corr_hv[:,0], perm=[0,2,1]),
                                 tf.transpose(inp_corr_hv[:,2], perm=[0,2,1]),
                                -tf.transpose(inp_corr_hv[:,3], perm=[0,2,1])], axis=1, name = "inp_corr_hvt")
#        return td, [inp_corr, inp_corr_h, inp_corr_v, inp_corr_hv, inp_corr_t, inp_corr_ht, inp_corr_vt, inp_corr_hvt]
        """
        return [tf.concat([tf.reshape(inp_corr,    [inp_corr.shape[0],-1]),td], axis=1,name = "out_corr"),
                tf.concat([tf.reshape(inp_corr_h,  [inp_corr.shape[0],-1]),td], axis=1,name = "out_corr_h"),
                tf.concat([tf.reshape(inp_corr_v,  [inp_corr.shape[0],-1]),td], axis=1,name = "out_corr_v"),
                tf.concat([tf.reshape(inp_corr_hv, [inp_corr.shape[0],-1]),td], axis=1,name = "out_corr_hv"),
                tf.concat([tf.reshape(inp_corr_t,  [inp_corr.shape[0],-1]),td], axis=1,name = "out_corr_t"),
                tf.concat([tf.reshape(inp_corr_ht, [inp_corr.shape[0],-1]),td], axis=1,name = "out_corr_ht"),
                tf.concat([tf.reshape(inp_corr_vt, [inp_corr.shape[0],-1]),td], axis=1,name = "out_corr_vt"),
                tf.concat([tf.reshape(inp_corr_hvt,[inp_corr.shape[0],-1]),td], axis=1,name = "out_corr_hvt")]
        """
        cl = 4 * TILE_SIDE * TILE_SIDE
        return [tf.concat([tf.reshape(inp_corr,    [-1,cl]),td], axis=1,name = "out_corr"),
                tf.concat([tf.reshape(inp_corr_h,  [-1,cl]),td], axis=1,name = "out_corr_h"),
                tf.concat([tf.reshape(inp_corr_v,  [-1,cl]),td], axis=1,name = "out_corr_v"),
                tf.concat([tf.reshape(inp_corr_hv, [-1,cl]),td], axis=1,name = "out_corr_hv"),
                tf.concat([tf.reshape(inp_corr_t,  [-1,cl]),td], axis=1,name = "out_corr_t"),
                tf.concat([tf.reshape(inp_corr_ht, [-1,cl]),td], axis=1,name = "out_corr_ht"),
                tf.concat([tf.reshape(inp_corr_vt, [-1,cl]),td], axis=1,name = "out_corr_vt"),
                tf.concat([tf.reshape(inp_corr_hvt,[-1,cl]),td], axis=1,name = "out_corr_hvt")]
#                           inp_corr_h, inp_corr_v, inp_corr_hv, inp_corr_t, inp_corr_ht, inp_corr_vt, inp_corr_hvt]
    

def network_sub(input, layout, reuse, sym8 = False):
    last_indx = None;
    fc = []
    for i, num_outs in enumerate (layout):
        if num_outs:
           if fc:
               inp = fc[-1]
               fc.append(slim.fully_connected(inp,    num_outs, activation_fn=lrelu, scope='g_fc_sub'+str(i), reuse = reuse))
           else:
               inp = input
               if sym8:
                   inp8 = sym_inputs8(inp)
                   num_non_sum = num_outs %  len(inp8) # if number of first layer outputs is not multiple of 8
                   num_sym8 =    num_outs // len(inp8) # number of symmetrical groups
                   fc_sym = []
                   for j in range (len(inp8)): # ==8
                       fc_sym.append(slim.fully_connected(inp8[j],    num_sym8, activation_fn=lrelu, scope='g_fc_sub'+str(i),     reuse = reuse | (j > 0)))
                   if num_non_sum > 0:
                       fc_sym.append(slim.fully_connected(inp,     num_non_sum, activation_fn=lrelu, scope='g_fc_sub'+str(i)+"r", reuse = reuse))    
                   fc.append(tf.concat(fc_sym, 1, name='sym_input_layer'))
               else:
                   fc.append(slim.fully_connected(inp,    num_outs, activation_fn=lrelu, scope='g_fc_sub'+str(i), reuse = reuse))
           
    return fc[-1]

def network_inter(input, layout):
    last_indx = None;
    fc = []
    for i, num_outs in enumerate (layout):
        if num_outs:
           if fc:
               inp = fc[-1]
           else:
               inp = input
           fc.append(slim.fully_connected(inp,    num_outs, activation_fn=lrelu, scope='g_fc_inter'+str(i)))
    if USE_CONFIDENCE:
        fc_out  = slim.fully_connected(fc[-1],     2, activation_fn=lrelu,scope='g_fc_inter_out')
    else:     
        fc_out  = slim.fully_connected(fc[-1],     1, activation_fn=None,scope='g_fc_inter_out')
        #If using residual disparity, split last layer into 2 or remove activation and add rectifier to confidence only  
    return fc_out

def network_siam(input, # now [?,9,325]-> [?,25,325]
                 layout1, 
                 layout2,
                 sym8 =        False,
                 only_tile =   None): # just for debugging - feed only data from the center sub-network
    with tf.name_scope("Siam_net"):
        num_legs =  input.shape[1] # == 9
        inter_list = []
        reuse = False
        for i in range (num_legs):
            if (only_tile is None) or (i == only_tile):
                inter_list.append(network_sub(input[:,i,:],
                                          layout= layout1,
                                          reuse= reuse,
                                          sym8 = sym8))
                reuse = True
        inter_tensor = tf.concat(inter_list, 1, name='inter_tensor')
        return  network_inter (inter_tensor, layout2)   

def debug_gt_variance(
        indx,        # This tile index (0..8)
        center_indx, # center tile index
        gt_ds_batch # [?:9:2]
        ):
    with tf.name_scope("Debug_GT_Variance"):
        tf_num_tiles =       tf.shape(gt_ds_batch)[0]
        d_gt_this =   tf.reshape(gt_ds_batch[:,2 * indx],[-1],                     name = "d_this")
        d_gt_center = tf.reshape(gt_ds_batch[:,2 * center_indx],[-1],              name = "d_center")
        d_gt_diff =   tf.subtract(d_gt_this, d_gt_center,                          name = "d_diff")
        d_gt_diff2 =  tf.multiply(d_gt_diff, d_gt_diff,                            name = "d_diff2")
        d_gt_var =    tf.reduce_mean(d_gt_diff2,                                   name = "d_gt_var")
        return  d_gt_var
    
    
def batchLoss(out_batch,                   # [batch_size,(1..2)] tf_result
              target_disparity_batch,      # [batch_size]        tf placeholder
              gt_ds_batch,                 # [batch_size,2]      tf placeholder
              absolute_disparity =     True, #when false there should be no activation on disparity output ! 
              use_confidence =         True, 
              lambda_conf_avg =        0.01,
              lambda_conf_pwr =        0.1,
              conf_pwr =               2.0,
              gt_conf_offset =         0.08,
              gt_conf_pwr =            1.0,
              error2_offset =          0.0025, # 0.0, # 0.0025, # (0.05^2) ~= coring
              disp_wmin =              1.0,    # minimal disparity to apply weight boosting for small disparities
              disp_wmax =              8.0,    # maximal disparity to apply weight boosting for small disparities
              use_out =                False):  # use calculated disparity for disparity weight boosting (False - use target disparity)
               
    with tf.name_scope("BatchLoss"):
        """
        Here confidence should be after relU. Disparity - may be also if absolute, but no activation if output is residual disparity
        """
        tf_lambda_conf_avg = tf.constant(lambda_conf_avg, dtype=tf.float32, name="tf_lambda_conf_avg")
        tf_lambda_conf_pwr = tf.constant(lambda_conf_pwr, dtype=tf.float32, name="tf_lambda_conf_pwr")
        tf_conf_pwr =        tf.constant(conf_pwr,        dtype=tf.float32, name="tf_conf_pwr")
        tf_gt_conf_offset =  tf.constant(gt_conf_offset,  dtype=tf.float32, name="tf_gt_conf_offset")
        tf_gt_conf_pwr =     tf.constant(gt_conf_pwr,     dtype=tf.float32, name="tf_gt_conf_pwr")
        tf_num_tiles =       tf.shape(gt_ds_batch)[0]
        tf_0f =              tf.constant(0.0,             dtype=tf.float32, name="tf_0f")
        tf_1f =              tf.constant(1.0,             dtype=tf.float32, name="tf_1f")
        tf_maxw =            tf.constant(1.0,             dtype=tf.float32, name="tf_maxw")
        if gt_conf_pwr == 0:
            w = tf.ones((out_batch.shape[0]), dtype=tf.float32,name="w_ones")
        else:
    #        w_slice = tf.slice(gt_ds_batch,[0,1],[-1,1],              name = "w_gt_slice")
            w_slice = tf.reshape(gt_ds_batch[:,1],[-1],                     name = "w_gt_slice")
            
            w_sub =   tf.subtract      (w_slice, tf_gt_conf_offset,         name = "w_sub")
    #        w_clip =  tf.clip_by_value(w_sub, tf_0f,tf_maxw,              name = "w_clip")
            w_clip =  tf.maximum(w_sub, tf_0f,                              name = "w_clip")
            if gt_conf_pwr == 1.0:
                w = w_clip
            else:
                w=tf.pow(w_clip, tf_gt_conf_pwr, name = "w_pow")
    
        if use_confidence:
            tf_num_tilesf =      tf.cast(tf_num_tiles, dtype=tf.float32,     name="tf_num_tilesf")
    #        conf_slice =     tf.slice(out_batch,[0,1],[-1,1],                name = "conf_slice")
            conf_slice =     tf.reshape(out_batch[:,1],[-1],                 name = "conf_slice")
            conf_sum =       tf.reduce_sum(conf_slice,                       name = "conf_sum")
            conf_avg =       tf.divide(conf_sum, tf_num_tilesf,              name = "conf_avg")
            conf_avg1 =      tf.subtract(conf_avg, tf_1f,                    name = "conf_avg1")
            conf_avg2 =      tf.square(conf_avg1,                            name = "conf_avg2")
            cost2 =          tf.multiply (conf_avg2, tf_lambda_conf_avg,     name = "cost2")
    
            iconf_avg =      tf.divide(tf_1f, conf_avg,                      name = "iconf_avg")
            nconf =          tf.multiply (conf_slice, iconf_avg,             name = "nconf") #normalized confidence
            nconf_pwr =      tf.pow(nconf, conf_pwr,                         name = "nconf_pwr")
            nconf_pwr_sum =  tf.reduce_sum(nconf_pwr,                        name = "nconf_pwr_sum")
            nconf_pwr_offs = tf.subtract(nconf_pwr_sum, tf_1f,               name = "nconf_pwr_offs")
            cost3 =          tf.multiply (conf_avg2, nconf_pwr_offs,         name = "cost3")
            w_all =          tf.multiply (w, nconf,                          name = "w_all")
        else:
            w_all = w
#            cost2 = 0.0
#            cost3 = 0.0    
        # normalize weights
        w_sum =              tf.reduce_sum(w_all,                            name = "w_sum")
        iw_sum =             tf.divide(tf_1f, w_sum,                         name = "iw_sum")
        w_norm =             tf.multiply (w_all, iw_sum,                     name = "w_norm")
        
        disp_slice =         tf.reshape(out_batch[:,0],[-1],                 name = "disp_slice")
        d_gt_slice =         tf.reshape(gt_ds_batch[:,0],[-1],               name = "d_gt_slice")
        
        td_flat =        tf.reshape(target_disparity_batch,[-1],         name = "td_flat")
        if absolute_disparity:
            adisp =          disp_slice
        else:
            adisp =          tf.add(disp_slice, td_flat,                     name = "adisp")
        out_diff =           tf.subtract(adisp, d_gt_slice,                  name = "out_diff")
            
            
        out_diff2 =          tf.square(out_diff,                             name = "out_diff2")
        out_wdiff2 =         tf.multiply (out_diff2, w_norm,                 name = "out_wdiff2")
        
        cost1 =              tf.reduce_sum(out_wdiff2,                       name = "cost1")
        
        out_diff2_offset =   tf.subtract(out_diff2, error2_offset,           name = "out_diff2_offset")
        out_diff2_biased =   tf.maximum(out_diff2_offset, 0.0,               name = "out_diff2_biased")
        
        # calculate disparity-based weight boost
        if use_out:
            dispw =          tf.clip_by_value(adisp, disp_wmin, disp_wmax,   name = "dispw")
        else:
            dispw =          tf.clip_by_value(td_flat, disp_wmin, disp_wmax, name = "dispw")
        dispw_boost =        tf.divide(disp_wmax, dispw,                     name = "dispw_boost")
        dispw_comp =         tf.multiply (dispw_boost, w_norm,               name = "dispw_comp") #HERE??
#        dispw_comp =         tf.multiply (w_norm, w_norm,               name = "dispw_comp") #HERE??
        dispw_sum =          tf.reduce_sum(dispw_comp,                       name = "dispw_sum")
        idispw_sum =         tf.divide(tf_1f, dispw_sum,                     name = "idispw_sum")
        dispw_norm =         tf.multiply (dispw_comp, idispw_sum,            name = "dispw_norm")
        
        out_diff2_wbiased =  tf.multiply(out_diff2_biased, dispw_norm,       name = "out_diff2_wbiased")
#        out_diff2_wbiased =  tf.multiply(out_diff2_biased, w_norm,       name = "out_diff2_wbiased")
        cost1b =             tf.reduce_sum(out_diff2_wbiased,                name = "cost1b")
        
        if use_confidence:
            cost12 =         tf.add(cost1b, cost2,                           name = "cost12")
            cost123 =        tf.add(cost12, cost3,                           name = "cost123")    
            
            return cost123, disp_slice, d_gt_slice, out_diff,out_diff2, w_norm, out_wdiff2, cost1
        else:
            return cost1b,  disp_slice, d_gt_slice, out_diff,out_diff2, w_norm, out_wdiff2, cost1
    
corr2d_Nx325 = tf.concat([tf.reshape(next_element_tt['corr2d'],[-1,cluster_size,FEATURES_PER_TILE], name="coor2d_cluster"),
                          tf.reshape(next_element_tt['target_disparity'], [-1,cluster_size, 1], name="targdisp_cluster")], axis=2, name = "corr2d_Nx325")

out =       network_siam(input=corr2d_Nx325,
                         layout1 =   NN_LAYOUT1, 
                         layout2 =   NN_LAYOUT2,
                         sym8 =      SYM8_SUB,
                         only_tile = ONLY_TILE) #Remove/put None for normal operation
#            w_slice = tf.reshape(gt_ds_batch[:,1],[-1],                     name = "w_gt_slice")

# Extract target disparity and GT corresponding to the center tile (reshape - just to name)
#target_disparity_batch_center = tf.reshape(next_element_tt['target_disparity'][:,center_tile_index:center_tile_index+1] , [-1,1],  name = "target_center")
#gt_ds_batch_center =            tf.reshape(next_element_tt['gt_ds'][:,2 * center_tile_index: 2 * center_tile_index+1],    [-1,2],  name = "gt_ds_center")

G_loss, _disp_slice, _d_gt_slice, _out_diff, _out_diff2, _w_norm, _out_wdiff2, _cost1 = batchLoss(out_batch =         out,        # [batch_size,(1..2)] tf_result
              target_disparity_batch=  next_element_tt['target_disparity'][:,center_tile_index:center_tile_index+1], # target_disparity_batch_center, # next_element_tt['target_disparity'], # target_disparity, ### target_d,   # [batch_size]        tf placeholder
              gt_ds_batch =            next_element_tt['gt_ds'][:,2 * center_tile_index: 2 * (center_tile_index +1)],  # gt_ds_batch_center, ## next_element_tt['gt_ds'], # gt_ds, ### gt,         # [batch_size,2]      tf placeholder
              absolute_disparity =     ABSOLUTE_DISPARITY,
              use_confidence =         USE_CONFIDENCE, # True, 
              lambda_conf_avg =        0.01,
              lambda_conf_pwr =        0.1,
              conf_pwr =               2.0,
              gt_conf_offset =         0.08,
              gt_conf_pwr =            2.0,
              error2_offset =          0, # 0.0025, # (0.05^2)
              disp_wmin =              1.0,    # minimal disparity to apply weight boosting for small disparities
              disp_wmax =              8.0,    # maximal disparity to apply weight boosting for small disparities
              use_out =                False)  # use calculated disparity for disparity weight boosting (False - use target disparity)
              
#debug
GT_variance =  debug_gt_variance(indx = 0,        # This tile index (0..8)
                                 center_indx = 4, # center tile index
                                 gt_ds_batch = next_element_tt['gt_ds'])# [?:18]
              
tf_ph_G_loss = tf.placeholder(tf.float32,shape=None,name='G_loss_avg')
tf_ph_sq_diff = tf.placeholder(tf.float32,shape=None,name='sq_diff_avg')
tf_gtvar_diff = tf.placeholder(tf.float32,shape=None,name='gtvar_diff')
tf_img_test0 = tf.placeholder(tf.float32,shape=None,name='img_test0')
tf_img_test9 = tf.placeholder(tf.float32,shape=None,name='img_test9')
with tf.name_scope('sample'):
    tf.summary.scalar("G_loss",       G_loss)
    tf.summary.scalar("sq_diff",      _cost1)
    tf.summary.scalar("gtvar_diff",   GT_variance)
    
with tf.name_scope('epoch_average'):
    tf.summary.scalar("G_loss_epoch", tf_ph_G_loss)
    tf.summary.scalar("sq_diff_epoch",tf_ph_sq_diff)
    tf.summary.scalar("gtvar_diff",   tf_gtvar_diff)
    
    tf.summary.scalar("img_test0",   tf_img_test0)
    tf.summary.scalar("img_test9",   tf_img_test9)

t_vars=tf.trainable_variables()
lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

saver=tf.train.Saver()

ROOT_PATH  = './attic/nn_ds_neibs6_graph'+SUFFIX+"/"
TRAIN_PATH =  ROOT_PATH + 'train'
TEST_PATH  =  ROOT_PATH + 'test'
TEST_PATH1  = ROOT_PATH + 'test1'

# CLEAN OLD STAFF
shutil.rmtree(TRAIN_PATH, ignore_errors=True)
shutil.rmtree(TEST_PATH, ignore_errors=True)
shutil.rmtree(TEST_PATH1, ignore_errors=True)

WIDTH=324
HEIGHT=242

with tf.Session()  as sess:
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    merged = tf.summary.merge_all()
    train_writer =    tf.summary.FileWriter(TRAIN_PATH, sess.graph)
    test_writer  =    tf.summary.FileWriter(TEST_PATH, sess.graph)
    test_writer1  =   tf.summary.FileWriter(TEST_PATH1, sess.graph)
    loss_train_hist=  np.empty(dataset_train_size, dtype=np.float32)
    loss_test_hist=   np.empty(dataset_test_size, dtype=np.float32)
    loss2_train_hist= np.empty(dataset_train_size, dtype=np.float32)
    loss2_test_hist=  np.empty(dataset_test_size, dtype=np.float32)
    train_avg = 0.0     
    train2_avg = 0.0
    test_avg = 0.0     
    test2_avg = 0.0
    gtvar_train_hist=  np.empty(dataset_train_size, dtype=np.float32)
    gtvar_test_hist=   np.empty(dataset_test_size, dtype=np.float32)
    gtvar_train = 0.0
    gtvar_test = 0.0
    gtvar_train_avg = 0.0
    gtvar_test_avg =  0.0
    img_gain_test0 =  1.0
    img_gain_test9 =  1.0
    
#    num_train_variants = len(files_train_lvar)
    num_train_variants = len(datasets_train)
    for epoch in range (EPOCHS_TO_RUN):
#        file_index = (epoch // 20) % 2 
        file_index = epoch  % num_train_variants
        if   epoch >=600:
            learning_rate = LR600
        elif epoch >=400:
            learning_rate = LR400
        elif epoch >=200:
            learning_rate = LR200
        elif epoch >=100:
            learning_rate = LR100
        else:
            learning_rate = LR
#        print ("sr1",file=sys.stderr,end=" ")
        if (file_index == 0) and SHUFFLE_FILES:
            print_time("Shuffling how datasets datasets_train_lvar and datasets_train_hvar are zipped together", end="")

            # datasets_train = zip_lvar_hvar(datasets_train_lvar, datasets_train_hvar, del_src= not SHUFFLE_FILES, shuffle = SHUFFLE_FILES)
            shuffle_in_place (datasets_train, 0, 2)
            shuffle_in_place (datasets_train, 1, 2)
            print_time("  Done")
            
        sess.run(iterator_tt.initializer, feed_dict={corr2d_train_placeholder:           datasets_train[file_index]['corr2d'],
                                                     target_disparity_train_placeholder: datasets_train[file_index]['target_disparity'],
                                                     gt_ds_train_placeholder:            datasets_train[file_index]['gt_ds']})
        for i in range(dataset_train_size):
            try:
                train_summary,_, G_loss_trained,  output, disp_slice, d_gt_slice, out_diff, out_diff2, w_norm, out_wdiff2, out_cost1, gt_variance  = sess.run(
                    [   merged,
                        G_opt,
                        G_loss,
                        out,
                        _disp_slice,
                        _d_gt_slice,
                        _out_diff,
                        _out_diff2,
                        _w_norm,
                        _out_wdiff2,
                        _cost1,
                        GT_variance
                    ],
                    feed_dict={lr:learning_rate,tf_ph_G_loss:train_avg, tf_ph_sq_diff:train2_avg, tf_gtvar_diff:gtvar_train_avg, tf_img_test0:img_gain_test0, tf_img_test9:img_gain_test9}) # previous value of *_avg
                
                loss_train_hist[i] =  G_loss_trained
                loss2_train_hist[i] = out_cost1
                gtvar_train_hist[i] = gt_variance
            except tf.errors.OutOfRangeError:
                print("train done at step %d"%(i))
                break

        train_avg =       np.average(loss_train_hist).astype(np.float32)     
        train2_avg =      np.average(loss2_train_hist).astype(np.float32)
        gtvar_train_avg = np.average(gtvar_train_hist).astype(np.float32)
        
        test_summaries = [0.0]*len(datasets_test)
        tst_avg =        [0.0]*len(datasets_test)
        tst2_avg =       [0.0]*len(datasets_test)
        for ntest,dataset_test in enumerate(datasets_test):
            sess.run(iterator_tt.initializer, feed_dict={corr2d_train_placeholder:      dataset_test['corr2d'],
                                                    target_disparity_train_placeholder: dataset_test['target_disparity'],
                                                    gt_ds_train_placeholder:            dataset_test['gt_ds']})
            for i in range(dataset_test_size):
                try:
                    test_summaries[ntest], G_loss_tested, output, disp_slice, d_gt_slice, out_diff, out_diff2, w_norm, out_wdiff2, out_cost1, gt_variance = sess.run(
                        [merged,
                         G_loss,
                         out,
                         _disp_slice,
                         _d_gt_slice,
                         _out_diff,
                         _out_diff2,
                         _w_norm,
                         _out_wdiff2,
                         _cost1,
                         GT_variance
                         ],
                            feed_dict={lr:learning_rate,tf_ph_G_loss:test_avg, tf_ph_sq_diff:test2_avg, tf_gtvar_diff:gtvar_test_avg, tf_img_test0:img_gain_test0, tf_img_test9:img_gain_test9})  # previous value of *_avg
                    loss_test_hist[i] =  G_loss_tested
                    loss2_test_hist[i] = out_cost1
                    gtvar_test_hist[i] = gt_variance
                except tf.errors.OutOfRangeError:
                    print("test done at step %d"%(i))
                    break
                    
            test_avg =  np.average(loss_test_hist).astype(np.float32)
            tst_avg[ntest] =  test_avg   
            test2_avg = np.average(loss2_test_hist).astype(np.float32)
            tst2_avg[ntest] =  test2_avg   
            gtvar_test_avg = np.average(gtvar_test_hist).astype(np.float32)
             
        train_writer.add_summary(train_summary, epoch)
        test_writer.add_summary(test_summaries[0], epoch)
        test_writer1.add_summary(test_summaries[1], epoch)
        
        print_time("%d:%d -> %f %f %f (%f %f %f) dbg:%f %f"%(epoch,i,train_avg, tst_avg[0], tst_avg[1], train2_avg, tst2_avg[0], tst2_avg[1], gtvar_train_avg, gtvar_test_avg))
        if ((epoch + 1) == EPOCHS_TO_RUN) or (((epoch + 1) % EPOCHS_FULL_TEST) == 0):
###################################################
# Read the full image
################################################### 
            test_summaries_img = [0.0]*len(datasets_img)
        #    disp_out=  np.empty((dataset_img_size * BATCH_SIZE), dtype=np.float32)
            disp_out=  np.empty((WIDTH*HEIGHT), dtype=np.float32)
            
            for ntest,dataset_img in enumerate(datasets_img):
                sess.run(iterator_tt.initializer, feed_dict={corr2d_train_placeholder:      dataset_img['corr2d'],
                                                        target_disparity_train_placeholder: dataset_img['target_disparity'],
                                                        gt_ds_train_placeholder:            dataset_img['gt_ds']})
                for start_offs in range(0,disp_out.shape[0],BATCH_SIZE):
                    end_offs = min(start_offs+BATCH_SIZE,disp_out.shape[0])
                    
                    try:
                        test_summaries_img[ntest], G_loss_tested, output, disp_slice, d_gt_slice, out_diff, out_diff2, w_norm, out_wdiff2, out_cost1, gt_variance = sess.run(
                            [merged,
                             G_loss,
                             out,
                             _disp_slice,
                             _d_gt_slice,
                             _out_diff,
                             _out_diff2,
                             _w_norm,
                             _out_wdiff2,
                             _cost1,
                             GT_variance
                             ],
                                feed_dict={lr:learning_rate,tf_ph_G_loss:test_avg, tf_ph_sq_diff:test2_avg, tf_gtvar_diff:gtvar_test_avg, tf_img_test0:img_gain_test0, tf_img_test9:img_gain_test9})  # previous value of *_avg
                    except tf.errors.OutOfRangeError:
                        print("test done at step %d"%(i))
                        break
                    try:
                        disp_out[start_offs:end_offs] = output.flatten()
                    except ValueError:
                        print("dataset_img_size= %d, i=%d, output.shape[0]=%d "%(dataset_img_size, i, output.shape[0]))
                        break;    
                    pass
                result_file = result_files[ntest]
                try:
                    os.makedirs(os.path.dirname(result_file))
                except:
                    pass     

                rslt = np.concatenate([disp_out.reshape(-1,1), t_disp, gtruth],1)
                np.save(result_file,           rslt.reshape(HEIGHT,WIDTH,-1))
                
                rslt = eval_results(result_file, ABSOLUTE_DISPARITY,radius=CLUSTER_RADIUS)                
                img_gain_test0 = rslt[0][0]/rslt[0][1]   
                img_gain_test9 = rslt[9][0]/rslt[9][1]   
     
     # Close writers
    train_writer.close()
    test_writer.close()
    test_writer1.close()
#reports error: Exception ignored in: <bound method BaseSession.__del__ of <tensorflow.python.client.session.Session object at 0x7efc5f720ef0>> if there is no print before exit()

print("All done")
exit (0)
