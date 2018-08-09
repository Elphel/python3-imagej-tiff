#!/usr/bin/env python3
from numpy import float64

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

TIME_START = time.time()
TIME_LAST  = TIME_START
DEBUG_LEVEL= 1
DISP_BATCH_BINS =   20 # Number of batch disparity bins
STR_BATCH_BINS =    10 # Number of batch strength bins
FILES_PER_SCENE =    5 # number of random offset files for the scene to select from (0 - use all available)
#MIN_BATCH_CHOICES = 10 # minimal number of tiles in a file for each bin to select from 
#MAX_BATCH_FILES =   10 #maximal number of files to use in a batch
MAX_EPOCH =        500
LR =               1e-4 # learning rate
LR100 =            1e-4
USE_CONFIDENCE =     False
ABSOLUTE_DISPARITY = True # False # True # False
DEBUG_PLT_LOSS =     True
TILE_LAYERS =        4
FILE_TILE_SIDE =     9
TILE_SIDE =          9 # 7
TILE_SIZE =         TILE_SIDE* TILE_SIDE # == 81
FEATURES_PER_TILE =  TILE_LAYERS * TILE_SIZE# == 324
EPOCHS_TO_RUN =     10000 #0
EPOCHS_SAME_FILE =   20
RUN_TOT_AVG =       100 # last batches to average. Epoch is 307 training  batches  
#BATCH_SIZE =       1080//9 # == 120 Each batch of tiles has balanced D/S tiles, shuffled batches but not inside batches
BATCH_SIZE =       2*1080//9 # == 120 Each batch of tiles has balanced D/S tiles, shuffled batches but not inside batches
SHUFFLE_EPOCH =    True
NET_ARCH1 =          4# 3 # overwrite with argv?
NET_ARCH2 =          0# 3 # overwrite with argv?
ONLY_TILE =          None # 0 # 4# None # (remove all but center tile data), put None here for normal operation)


#DEBUG_PACK_TILES = True
SUFFIX=str(NET_ARCH1)+'-'+str(NET_ARCH2)+ (["R","A"][ABSOLUTE_DISPARITY])
# CLUSTER_RADIUS should match input data
CLUSTER_RADIUS =     1 # 1 - 3x3, 2 - 5x5 tiles

NN_LAYOUTS = {0:[0,   0,   0,   32,  20,  16],
              1:[0,   0,   0,  256, 128,  64],
              2:[0, 128,  32,   32,  32,  16],
              3:[0,   0,  40,   32,  20,  16],
              4:[0,   0,   0,   0,   16,  16]
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
#    dataset = tf.data.TFRecordDataset(filenames)
    if not  '.tfrecords' in train_filename:
        train_filename += '.tfrecords'
    record_iterator = tf.python_io.tf_record_iterator(path=train_filename)
    corr2d_list=[]
    target_disparity_list=[]
    gt_ds_list = []
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        corr2d_list.append           (np.array(example.features.feature['corr2d'].float_list.value, dtype=np.float32))
#        target_disparity_list.append(np.array(example.features.feature['target_disparity'].float_list.value[0], dtype=np.float32))
        target_disparity_list.append (np.array(example.features.feature['target_disparity'].float_list.value, dtype=np.float32))
        gt_ds_list.append            (np.array(example.features.feature['gt_ds'].float_list.value, dtype= np.float32))
        pass
    corr2d=            np.array(corr2d_list)
    target_disparity = np.array(target_disparity_list)
    gt_ds =            np.array(gt_ds_list)
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
def reformat_to_clusters(datasets_data):
    cluster_size = (2 * CLUSTER_RADIUS + 1) * (2 * CLUSTER_RADIUS + 1)
# Reformat input data
    for rec in datasets_data:
        rec['corr2d'] =           rec['corr2d'].reshape(          (rec['corr2d'].shape[0]//cluster_size,           rec['corr2d'].shape[1] * cluster_size)) 
        rec['target_disparity'] = rec['target_disparity'].reshape((rec['target_disparity'].shape[0]//cluster_size, rec['target_disparity'].shape[1] * cluster_size)) 
        rec['gt_ds'] =            rec['gt_ds'].reshape(           (rec['gt_ds'].shape[0]//cluster_size,            rec['gt_ds'].shape[1] * cluster_size))

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

"""
Start of the main code
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

files_train_lvar = ["/home/eyesis/x3d_data/data_sets/tf_data_3x3a/train000_R1_LE_1.5.tfrecords"]
"""
                    "/home/eyesis/x3d_data/data_sets/tf_data_3x3/train-000_R1_LE_1.5.tfrecords",
                    "/home/eyesis/x3d_data/data_sets/tf_data_3x3/train-001_R1_LE_1.5.tfrecords",
                    "/home/eyesis/x3d_data/data_sets/tf_data_3x3/train-002_R1_LE_1.5.tfrecords",
                    "/home/eyesis/x3d_data/data_sets/tf_data_3x3/train-003_R1_LE_1.5.tfrecords",
                    "/home/eyesis/x3d_data/data_sets/tf_data_3x3/train-004_R1_LE_1.5.tfrecords",
                    "/home/eyesis/x3d_data/data_sets/tf_data_3x3/train-005_R1_LE_1.5.tfrecords",
                    "/home/eyesis/x3d_data/data_sets/tf_data_3x3/train-006_R1_LE_1.5.tfrecords",
                    "/home/eyesis/x3d_data/data_sets/tf_data_3x3/train-007_R1_LE_1.5.tfrecords"]
"""
#files_train_hvar = ["/home/eyesis/x3d_data/data_sets/tf_data_3x3/train-000_R1_LE_1.5.tfrecords",
#                    "/home/eyesis/x3d_data/data_sets/tf_data_3x3/train-001_R1_LE_1.5.tfrecords"]
files_train_hvar = []

#file_test_lvar=     "/home/eyesis/x3d_data/data_sets/tf_data_3x3/test-TEST_R1_LE_1.5.tfrecords" # "/home/eyesis/x3d_data/data_sets/train-000_R1_LE_1.5.tfrecords"
file_test_lvar=     "/home/eyesis/x3d_data/data_sets/tf_data_3x3a/train000_R1_LE_1.5.tfrecords" # "/home/eyesis/x3d_data/data_sets/train-000_R1_LE_1.5.tfrecords"
file_test_hvar=     None # "/home/eyesis/x3d_data/data_sets/tf_data_3x3/train-002_R1_LE_1.5.tfrecords" # "/home/eyesis/x3d_data/data_sets/train-000_R1_LE_1.5.tfrecords"
weight_hvar = 0.13
weight_lvar = 1.0 - weight_hvar 


import tensorflow as tf
import tensorflow.contrib.slim as slim

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
#CLUSTER_RADIUS
cluster_size = (2 * CLUSTER_RADIUS + 1) * (2 * CLUSTER_RADIUS + 1)
center_tile_index = 2 * CLUSTER_RADIUS * (CLUSTER_RADIUS + 1)
# Reformat input data
if FILE_TILE_SIDE > TILE_SIDE:
    print_time("Reducing correlation tile size from %d to %d"%(FILE_TILE_SIDE, TILE_SIDE), end="")
    reduce_tile_size(datasets_train_lvar, TILE_LAYERS, TILE_SIDE)
    reduce_tile_size(datasets_train_hvar, TILE_LAYERS, TILE_SIDE)
    if (file_test_lvar):
        reduce_tile_size([dataset_test_lvar], TILE_LAYERS, TILE_SIDE)
    if (file_test_hvar):
        reduce_tile_size([dataset_test_hvar], TILE_LAYERS, TILE_SIDE)
    print_time("  Done")
pass
pass
print_time("Reshaping train data (low variance)", end="")
reformat_to_clusters(datasets_train_lvar)
print_time("  Done")
print_time("Reshaping train data (high variance)", end="")
reformat_to_clusters(datasets_train_hvar)
print_time("  Done")
if (file_test_lvar):
    print_time("Reshaping test data (low variance)", end="")
    reformat_to_clusters([dataset_test_lvar])
    print_time("  Done")
if (file_test_hvar):    
    print_time("Reshaping test data (high variance)", end="")
    reformat_to_clusters([dataset_test_hvar])
    print_time("  Done")
pass    
    
    
"""
corr2d_trains =           [corr2d_train,            corr2d_train1] 
target_disparity_trains = [target_disparity_train,  target_disparity_train1] 
gt_ds_trains =            [gt_ds_train,             gt_ds_train1] 

corr2d_train_placeholder =           tf.placeholder(corr2d_train.dtype,           (None,FEATURES_PER_TILE)) # corr2d_train.shape)
target_disparity_train_placeholder = tf.placeholder(target_disparity_train.dtype, (None,1))  #target_disparity_train.shape)
gt_ds_train_placeholder =            tf.placeholder(gt_ds_train.dtype,            (None,2)) #gt_ds_train.shape)
"""
corr2d_train_placeholder =           tf.placeholder(datasets_train_lvar[0]['corr2d'].dtype,           (None,FEATURES_PER_TILE * cluster_size)) # corr2d_train.shape)
target_disparity_train_placeholder = tf.placeholder(datasets_train_lvar[0]['target_disparity'].dtype, (None,1 *   cluster_size))  #target_disparity_train.shape)
gt_ds_train_placeholder =            tf.placeholder(datasets_train_lvar[0]['gt_ds'].dtype,            (None,2 *   cluster_size)) #gt_ds_train.shape)

dataset_tt = tf.data.Dataset.from_tensor_slices({
    "corr2d":corr2d_train_placeholder,
    "target_disparity": target_disparity_train_placeholder,
    "gt_ds": gt_ds_train_placeholder})


#dataset_train_size = len(corr2d_train)

dataset_train_size = len(datasets_train_lvar[0]['corr2d'])
dataset_train_size //= BATCH_SIZE
dataset_test_size = len(dataset_test_lvar['corr2d'])
dataset_test_size //= BATCH_SIZE

#print_time("dataset_tt.output_types "+str(dataset_train.output_types)+", dataset_train.output_shapes "+str(dataset_train.output_shapes)+", number of elements="+str(dataset_train_size))
dataset_tt = dataset_tt.batch(BATCH_SIZE)
iterator_tt = dataset_tt.make_initializable_iterator()
next_element_tt = iterator_tt.get_next()
#print("dataset_tt.output_types "+str(dataset_tt.output_types)+", dataset_tt.output_shapes "+str(dataset_tt.output_shapes)+", number of elements="+str(dataset_train_size))


"""
iterator_test =  dataset_test.make_initializable_iterator()
next_element_test =  iterator_test.get_next()
"""
#https://www.tensorflow.org/versions/r1.5/programmers_guide/datasets

result_dir = './attic/result_neibs_'+     SUFFIX+'/'
checkpoint_dir = './attic/result_neibs_'+ SUFFIX+'/'
save_freq = 500

def lrelu(x):
    return tf.maximum(x*0.2,x)
#    return tf.nn.relu(x)

def network_fc_simple(input, arch = 0):
    layouts = {0:[0,   0,   0,   32,  20,  16],
               1:[0,   0,   0,  256, 128,  64],
               2:[0, 128,  32,   32,  32,  16],
               3:[0,   0,  40,   32,  20,  16]}
    layout = layouts[arch]
    last_indx = None;
    fc = []
    for i, num_outs in enumerate (layout):
        if num_outs:
           if fc:
               inp = fc[-1]
           else:
               inp = input
           fc.append(slim.fully_connected(inp, num_outs, activation_fn=lrelu,scope='g_fc'+str(i)))    
    if USE_CONFIDENCE:
        fc_out  = slim.fully_connected(fc[-1],     2, activation_fn=lrelu,scope='g_fc_out')
    else:     
        fc_out  = slim.fully_connected(fc[-1],     1, activation_fn=None,scope='g_fc_out')
        #If using residual disparity, split last layer into 2 or remove activation and add rectifier to confidence only  
    return fc_out
    """
    layouts = {0:[0,   0,   0,   32,  20,  16],
               1:[0,   0,   0,  256, 128,  64],
               2:[0, 128,  32,   32,  32,  16],
               3:[0,   0,  40,   32,  20,  16]}
    layout = layouts[arch]
    """

def network_sub(input, layout, reuse):
    last_indx = None;
    fc = []
    for i, num_outs in enumerate (layout):
        if num_outs:
           if fc:
               inp = fc[-1]
           else:
               inp = input
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

def network_siam(input, # now [?:9,325]
                 layout1, 
                 layout2,
                 only_tile=None): # just for debugging - feed only data from the center sub-network
    with tf.name_scope("Siam_net"):
        num_legs =  input.shape[1] # == 9
        inter_list = []
        reuse = False
        for i in range (num_legs):
            if (only_tile is None) or (i == only_tile):
                inter_list.append(network_sub(input[:,i,:],
                                          layout= layout1,
                                          reuse= reuse))
                reuse = True
        inter_tensor = tf.concat(inter_list, 1, name='inter_tensor')
        return  network_inter (inter_tensor, layout2)   
#corr2d9x325 = tf.concat([tf.reshape(next_element_tt['corr2d'],[None,cluster_size,FEATURES_PER_TILE]) , tf.reshape(next_element_tt['target_disparity'], [None,cluster_size, 1])],2)

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
              error2_offset =          0.0025, # (0.05^2)
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
        
    #    disp_slice =         tf.slice(out_batch,[0,0],[-1,1],                name = "disp_slice")
    #    d_gt_slice =         tf.slice(gt_ds_batch,[0,0],[-1,1],              name = "d_gt_slice")
        disp_slice =         tf.reshape(out_batch[:,0],[-1],                 name = "disp_slice")
        d_gt_slice =         tf.reshape(gt_ds_batch[:,0],[-1],               name = "d_gt_slice")
        
        """
        if absolute_disparity:
            out_diff =       tf.subtract(disp_slice, d_gt_slice,             name = "out_diff")
        else:
            td_flat =        tf.reshape(target_disparity_batch,[-1],         name = "td_flat")
            residual_disp =  tf.subtract(d_gt_slice, td_flat,                name = "residual_disp")
            out_diff =       tf.subtract(disp_slice, residual_disp,          name = "out_diff")
        """    
        td_flat =        tf.reshape(target_disparity_batch,[-1],         name = "td_flat")
        if absolute_disparity:
            adisp =          disp_slice
        else:
#            td_flat =        tf.reshape(target_disparity_batch,[-1],         name = "td_flat")
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
    
#In GPU - reformat inputs

##corr2d325 = tf.concat([next_element_tt['corr2d'], next_element_tt['target_disparity']],1)

#Should have shape (?,9,325)
corr2d9x325 = tf.concat([tf.reshape(next_element_tt['corr2d'],[-1,cluster_size,FEATURES_PER_TILE]) , tf.reshape(next_element_tt['target_disparity'], [-1,cluster_size, 1])],2)

#corr2d9x324 = tf.reshape( next_element_tt['corr2d'],          [-1, cluster_size, FEATURES_PER_TILE], name = 'corr2d9x324')
#td9x1 =       tf.reshape(next_element_tt['target_disparity'], [-1, cluster_size, 1],    name = 'td9x1')
#corr2d9x325 = tf.concat([corr2d9x324 , td9x1],2, name = 'corr2d9x325')


#    in_features = tf.concat([corr2d,target_disparity],0)

#out =       network_fc_simple(input=corr2d325, arch = NET_ARCH1)
out =       network_siam(input=corr2d9x325,
                         layout1 =   NN_LAYOUT1, 
                         layout2 =   NN_LAYOUT2,
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
              error2_offset =          0.0025, # (0.05^2)
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
with tf.name_scope('sample'):
    tf.summary.scalar("G_loss",       G_loss)
    tf.summary.scalar("sq_diff",      _cost1)
    tf.summary.scalar("gtvar_diff",   GT_variance)
    
with tf.name_scope('epoch_average'):
    tf.summary.scalar("G_loss_epoch", tf_ph_G_loss)
    tf.summary.scalar("sq_diff_epoch",tf_ph_sq_diff)
    tf.summary.scalar("gtvar_diff",   tf_gtvar_diff)

t_vars=tf.trainable_variables()
lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

saver=tf.train.Saver()

ROOT_PATH  = './attic/nn_ds_neibs_graph'+SUFFIX+"/"
TRAIN_PATH = ROOT_PATH + 'train'
TEST_PATH  = ROOT_PATH + 'test'

# CLEAN OLD STAFF
shutil.rmtree(TRAIN_PATH, ignore_errors=True)
shutil.rmtree(TEST_PATH, ignore_errors=True)

with tf.Session()  as sess:
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    merged = tf.summary.merge_all()
    train_writer =    tf.summary.FileWriter(TRAIN_PATH, sess.graph)
    test_writer  =    tf.summary.FileWriter(TEST_PATH, sess.graph)
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
    
    num_train_variants = len(files_train_lvar)
    for epoch in range (EPOCHS_TO_RUN):
#        file_index = (epoch // 20) % 2 
        file_index = epoch  % num_train_variants
        learning_rate = [LR,LR100][epoch >=100] 
    #       if SHUFFLE_EPOCH:
    #        dataset_tt = dataset_tt.shuffle(buffer_size=10000)
        """
        sess.run(iterator_tt.initializer, feed_dict={corr2d_train_placeholder: corr2d_trains[file_index],
                                                        target_disparity_train_placeholder: target_disparity_trains[file_index],
                                                        gt_ds_train_placeholder: gt_ds_trains[file_index]})
        """
        sess.run(iterator_tt.initializer, feed_dict={corr2d_train_placeholder:           datasets_train_lvar[file_index]['corr2d'],
                                                     target_disparity_train_placeholder: datasets_train_lvar[file_index]['target_disparity'],
                                                     gt_ds_train_placeholder:            datasets_train_lvar[file_index]['gt_ds']})
        for i in range(dataset_train_size):
            try:
#                train_summary,_, G_loss_trained,  output, disp_slice, d_gt_slice, out_diff, out_diff2, w_norm, out_wdiff2, out_cost1, corr2d325_out  = sess.run(
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
#                        corr2d325,
                    ],
                    feed_dict={lr:learning_rate,tf_ph_G_loss:train_avg, tf_ph_sq_diff:train2_avg, tf_gtvar_diff:gtvar_train_avg}) # previous value of *_avg
                
                # save all for now as a test
                #train_writer.add_summary(summary, i)
                #train_writer.add_summary(train_summary, i)
                loss_train_hist[i] =  G_loss_trained
                loss2_train_hist[i] = out_cost1
                gtvar_train_hist[i] = gt_variance
            except tf.errors.OutOfRangeError:
                print("train done at step %d"%(i))
                break
        train_avg =       np.average(loss_train_hist).astype(np.float32)     
        train2_avg =      np.average(loss2_train_hist).astype(np.float32)
        gtvar_train_avg = np.average(gtvar_train_hist).astype(np.float32)
        
#        _,_=sess.run([tf_ph_G_loss,tf_ph_sq_diff],feed_dict={tf_ph_G_loss:train_avg, tf_ph_sq_diff:train2_avg})
#tf_ph_G_loss = tf.placeholder(tf.float32,shape=None,name='G_loss_avg')
#tf_ph_sq_diff = tf.placeholder(tf.float32,shape=None,name='sq_diff_avg')
        """     
        sess.run(iterator_tt.initializer, feed_dict={corr2d_train_placeholder: corr2d_test,
                                                        target_disparity_train_placeholder: target_disparity_test,
                                                        gt_ds_train_placeholder: gt_ds_test})
        """
        sess.run(iterator_tt.initializer, feed_dict={corr2d_train_placeholder:   dataset_test_lvar['corr2d'],
                                                target_disparity_train_placeholder: dataset_test_lvar['target_disparity'],
                                                gt_ds_train_placeholder:            dataset_test_lvar['gt_ds']})

        for i in range(dataset_test_size):
            try:
#                test_summary, G_loss_tested, output, disp_slice, d_gt_slice, out_diff, out_diff2, w_norm, out_wdiff2, out_cost1, corr2d325_out = sess.run(
                test_summary, G_loss_tested, output, disp_slice, d_gt_slice, out_diff, out_diff2, w_norm, out_wdiff2, out_cost1, gt_variance = sess.run(
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
#                     corr2d325,
                     ],
                        feed_dict={lr:learning_rate,tf_ph_G_loss:test_avg, tf_ph_sq_diff:test2_avg, tf_gtvar_diff:gtvar_test_avg})  # previous value of *_avg
                loss_test_hist[i] =  G_loss_tested
                loss2_test_hist[i] = out_cost1
                gtvar_test_hist[i] = gt_variance
            except tf.errors.OutOfRangeError:
                print("test done at step %d"%(i))
                break
            
#            print_time("%d:%d -> %f"%(epoch,i,G_current))
        test_avg =  np.average(loss_test_hist).astype(np.float32)     
        test2_avg = np.average(loss2_test_hist).astype(np.float32)
        gtvar_test_avg = np.average(gtvar_test_hist).astype(np.float32)
             
#        _,_=sess.run([tf_ph_G_loss,tf_ph_sq_diff],feed_dict={tf_ph_G_loss:test_avg, tf_ph_sq_diff:test2_avg})
        
        train_writer.add_summary(train_summary, epoch)
        test_writer.add_summary(test_summary, epoch)
        
        print_time("%d:%d -> %f %f (%f %f) dbg:%f %f"%(epoch,i,train_avg, test_avg,train2_avg, test2_avg, gtvar_train_avg, gtvar_test_avg))
     
     # Close writers
    train_writer.close()
    test_writer.close()
#reports error: Exception ignored in: <bound method BaseSession.__del__ of <tensorflow.python.client.session.Session object at 0x7efc5f720ef0>> if there is no print before exit()

print("All done")
exit (0)
