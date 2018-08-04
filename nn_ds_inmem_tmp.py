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
LR =               1e-3 # learning rate
USE_CONFIDENCE =     False
ABSOLUTE_DISPARITY = False
DEBUG_PLT_LOSS =     True
FEATURES_PER_TILE =  324
EPOCHS_TO_RUN =     10000 #0
RUN_TOT_AVG =       100 # last batches to average. Epoch is 307 training  batches  
BATCH_SIZE =       1000 # Each batch of tiles has balanced D/S tiles, shuffled batches but not inside batches
SHUFFLE_EPOCH =    True
#DEBUG_PACK_TILES = True

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
        'corr2d':           tf.FixedLenFeature([324],tf.float32), #string),
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

#Main code
try:
    train_filenameTFR =  sys.argv[1]
except IndexError:
    train_filenameTFR = "/mnt/dde6f983-d149-435e-b4a2-88749245cc6c/home/eyesis/x3d_data/data_sets/tf_data/train.tfrecords"
try:
    test_filenameTFR =  sys.argv[2]
except IndexError:
    test_filenameTFR = "/mnt/dde6f983-d149-435e-b4a2-88749245cc6c/home/eyesis/x3d_data/data_sets/tf_data/test.tfrecords"
#FILES_PER_SCENE


print_time("Importing TensorCrawl")

import tensorflow as tf
import tensorflow.contrib.slim as slim

print_time("TensorCrawl imported")

print_time("Importing training data... ", end="")
corr2d_train, target_disparity_train, gt_ds_train = readTFRewcordsEpoch(train_filenameTFR)
print_time("  Done")
dataset_train = tf.data.Dataset.from_tensor_slices({
    "corr2d":corr2d_train,
    "target_disparity": target_disparity_train,
    "gt_ds": gt_ds_train})
dataset_train_size = len(corr2d_train)
print_time("dataset_train.output_types "+str(dataset_train.output_types)+", dataset_train.output_shapes "+str(dataset_train.output_shapes)+", number of elements="+str(dataset_train_size))
dataset_train = dataset_train.batch(BATCH_SIZE)
dataset_train_size /= BATCH_SIZE
print("dataset_train.output_types "+str(dataset_train.output_types)+", dataset_train.output_shapes "+str(dataset_train.output_shapes)+", number of elements="+str(dataset_train_size))
iterator_train = dataset_train.make_initializable_iterator()
next_element_train = iterator_train.get_next()

'''
print_time("Importing test data... ", end="")
corr2d_test, target_disparity_test, gt_ds_test = readTFRewcordsEpoch(test_filenameTFR)
print_time("  Done")
dataset_test =  tf.data.Dataset.from_tensor_slices({
    "corr2d":corr2d_test,
    "target_disparity": target_disparity_test,
    "gt_ds": gt_ds_test})
dataset_test_size = len(corr2d_test)
print_time("dataset_test.output_types "+str(dataset_test.output_types)+", dataset_test.output_shapes "+str(dataset_test.output_shapes)+", number of elements="+str(dataset_test_size))
dataset_test =  dataset_test.batch(BATCH_SIZE)
dataset_test_size /= BATCH_SIZE
print("dataset_test.output_types "+str(dataset_test.output_types)+", dataset_test.output_shapes "+str(dataset_test.output_shapes)+", number of elements="+str(dataset_test_size))
iterator_test =  dataset_test.make_initializable_iterator()
next_element_test =  iterator_test.get_next()
'''
#https://www.tensorflow.org/versions/r1.5/programmers_guide/datasets

result_dir = './attic/result_inmem/'
checkpoint_dir = './attic/result_inmem/'
save_freq = 500

def lrelu(x):
    return tf.maximum(x*0.2,x)
#    return tf.nn.relu(x)

def network(input):

#  fc1  = slim.fully_connected(input, 512, activation_fn=lrelu,scope='g_fc1')
#  fc2  = slim.fully_connected(fc1,   512, activation_fn=lrelu,scope='g_fc2')
  fc3  =     slim.fully_connected(input, 256, activation_fn=lrelu,scope='g_fc3')
  fc4  =     slim.fully_connected(fc3,   128, activation_fn=lrelu,scope='g_fc4')
  fc5  =     slim.fully_connected(fc4,    64, activation_fn=lrelu,scope='g_fc5')
  if USE_CONFIDENCE:
      fc6  = slim.fully_connected(fc5,     2, activation_fn=lrelu,scope='g_fc6')
  else:     
      fc6  = slim.fully_connected(fc5,     1, activation_fn=None,scope='g_fc6')
#If using residual disparity, split last layer into 2 or remove activation and add rectifier to confidence only  
  return fc6

def batchLoss(out_batch,                   # [batch_size,(1..2)] tf_result
              target_disparity_batch,      # [batch_size]        tf placeholder
              gt_ds_batch,                 # [batch_size,2]      tf placeholder
              absolute_disparity =     True, #when false there should be no activation on disparity output ! 
              use_confidence =         True, 
              lambda_conf_avg =        0.01,
              lambda_conf_pwr =        0.1,
              conf_pwr =               2.0,
              gt_conf_offset =         0.08,
              gt_conf_pwr =            1.0):
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
                w=tf.pow(w_clip, tf_gt_conf_pwr, name = "w")
    
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
        if absolute_disparity:
            out_diff =       tf.subtract(disp_slice, d_gt_slice,             name = "out_diff")
        else:
            td_flat =        tf.reshape(target_disparity_batch,[-1],         name = "td_flat")
            residual_disp =  tf.subtract(d_gt_slice, td_flat,                name = "residual_disp")
            out_diff =       tf.subtract(disp_slice, residual_disp,          name = "out_diff")
        out_diff2 =          tf.square(out_diff,                             name = "out_diff2")
        out_wdiff2 =         tf.multiply (out_diff2, w_norm,                 name = "out_wdiff2")
        
        cost1 =              tf.reduce_sum(out_wdiff2,                       name = "cost1")
        
        if use_confidence:
            cost12 =         tf.add(cost1,  cost2,                           name = "cost12")
            cost123 =        tf.add(cost12, cost3,                           name = "cost123")    
            
            return cost123, disp_slice, d_gt_slice, out_diff,out_diff2, w_norm, out_wdiff2, cost1
        else:
            return cost1, disp_slice, d_gt_slice, out_diff,out_diff2, w_norm, out_wdiff2, cost1
    

#corr2d325 = tf.concat([corr2d,target_disparity],0)
#corr2d325 = tf.concat([next_element_train['corr2d'],tf.reshape(next_element_train['target_disparity'],(-1,1))],1)
corr2d325 = tf.concat([next_element_train['corr2d'], next_element_train['target_disparity']],1)
#next_element_train

#    in_features = tf.concat([corr2d,target_disparity],0)

out =       network(corr2d325)
#Try standard loss functions first
G_loss, _disp_slice, _d_gt_slice, _out_diff, _out_diff2, _w_norm, _out_wdiff2, _cost1 = batchLoss(out_batch =         out,        # [batch_size,(1..2)] tf_result
              target_disparity_batch=  next_element_train['target_disparity'], # target_disparity, ### target_d,   # [batch_size]        tf placeholder
              gt_ds_batch =            next_element_train['gt_ds'], # gt_ds, ### gt,         # [batch_size,2]      tf placeholder
              absolute_disparity =     ABSOLUTE_DISPARITY,
              use_confidence =         USE_CONFIDENCE, # True, 
              lambda_conf_avg =        0.01,
              lambda_conf_pwr =        0.1,
              conf_pwr =               2.0,
              gt_conf_offset =         0.08,
              gt_conf_pwr =            1.0)

tf.summary.scalar("G_loss",G_loss)

t_vars=tf.trainable_variables()
lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

saver=tf.train.Saver()

ROOT_PATH  = './attic/nn_ds_inmem_graph1/'
TRAIN_PATH = ROOT_PATH + 'train'
TEST_PATH  = ROOT_PATH + 'test'

# CLEAN OLD STAFF
#shutil.rmtree(TRAIN_PATH)
#shutil.rmtree(TEST_PATH)

with tf.Session()  as sess:
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(TRAIN_PATH, sess.graph)
    test_writer  = tf.summary.FileWriter(TEST_PATH, sess.graph)
    
    for epoch in range(EPOCHS_TO_RUN):
        
        if SHUFFLE_EPOCH:
            dataset_train = dataset_train.shuffle(buffer_size=10000)
        sess.run(iterator_train.initializer)
        
        i=0
        while True:
            
            # overall are 307, start 'testing' testing from START_TEST
            START_TEST = 200
            
            # Train run
            if i<START_TEST:
                
                try:
    #                _, G_current,  output, disp_slice, d_gt_slice, out_diff, out_diff2, w_norm, out_wdiff2, out_cost1, corr2d325_out, target_disparity_out, gt_ds_out = sess.run(
                    train_summary,_, G_loss_trained,  output, disp_slice, d_gt_slice, out_diff, out_diff2, w_norm, out_wdiff2, out_cost1, corr2d325_out  = sess.run(
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
                            corr2d325,
    #                     target_disparity,
    #                     gt_ds
                        ],
                        feed_dict={lr:LR})
                    
                    # save all for now as a test
                    #train_writer.add_summary(summary, i)
                    #train_writer.add_summary(train_summary, i)
                    
                except tf.errors.OutOfRangeError:
                    break                
             
            # Test run
            else:

                try:
                    test_summary, G_loss_tested, output, disp_slice, d_gt_slice, out_diff, out_diff2, w_norm, out_wdiff2, out_cost1, corr2d325_out = sess.run(
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
                         corr2d325,
                         ],
                            feed_dict={lr:LR})
                    
                    #test_writer.add_summary(test_summary, i)
                    
                except tf.errors.OutOfRangeError:
                    break

            i+=1
            
#            print_time("%d:%d -> %f"%(epoch,i,G_current))
        train_writer.add_summary(train_summary, epoch)
        test_writer.add_summary(test_summary, epoch)
        
        print_time("%d:%d -> %f"%(epoch,i,G_loss_trained))
     
     # Close writers
    train_writer.close()
    test_writer.close()
#reports error: Exception ignored in: <bound method BaseSession.__del__ of <tensorflow.python.client.session.Session object at 0x7efc5f720ef0>> if there is no print before exit()

print("All done")
exit (0)
