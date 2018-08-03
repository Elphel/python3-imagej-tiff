#!/usr/bin/env python3
from numpy import float64

__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "andrey@elphel.com"


from PIL import Image

import os
import sys
import glob

import explore_data as exd
import pack_tile as pile

import numpy as np
import itertools

import time

import matplotlib.pyplot as plt

#http://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
TIME_START = time.time()
TIME_LAST  = TIME_START
DEBUG_LEVEL= 1
DISP_BATCH_BINS =   20 # Number of batch disparity bins
STR_BATCH_BINS =    10 # Number of batch strength bins
FILES_PER_SCENE =    5 # number of random offset files for the scene to select from (0 - use all available)
MIN_BATCH_CHOICES = 10 # minimal number of tiles in a file for each bin to select from 
MAX_BATCH_FILES =   10 #maximal number of files to use in a batch
MAX_EPOCH =        500
LR =               1e-4 # learning rate
USE_CONFIDENCE =     False
ABSOLUTE_DISPARITY = False
DEBUG_PLT_LOSS =   True
#DEBUG_PACK_TILES = True

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
    print(("%s"+bcolors.BOLDWHITE+"at %.4fs (+%.4fs)"+bcolors.ENDC)%(txt,t-TIME_START,t-TIME_LAST), end = end)
    TIME_LAST = t


#Main code
try:
    topdir_train = sys.argv[1]
except IndexError:
    topdir_train = "/mnt/dde6f983-d149-435e-b4a2-88749245cc6c/home/eyesis/x3d_data/data_sets/train"#test" #all/"
try:
    topdir_test = sys.argv[2]
except IndexError:
    topdir_test = "/mnt/dde6f983-d149-435e-b4a2-88749245cc6c/home/eyesis/x3d_data/data_sets/test"#test" #all/"

print_time("Exploring dataset (long operation)")
ex_data = exd.ExploreData(
             topdir_train =         topdir_train,
             topdir_test =          topdir_test,
             debug_level =          0, #DEBUG_LEVEL, # 3, ##0, #3,
             disparity_bins =     200, #1000,
             strength_bins =      100,
             disparity_min_drop =  -0.1,
             disparity_min_clip =  -0.1,
             disparity_max_drop =  20.0, #100.0,
             disparity_max_clip =  20.0, #100.0,
             strength_min_drop =    0.1,
             strength_min_clip =    0.1,
             strength_max_drop =    1.0,
             strength_max_clip =    0.9,
             hist_sigma =           2.0,  # Blur log histogram
             hist_cutoff=           0.001) #  of maximal  
print_time(("Done exploring dataset, assigning DSI histogram tiles to batch bins (%d disparity bins, %d strength bins, %d disparity offsets: total %d tiles per batch)"%(
    DISP_BATCH_BINS,STR_BATCH_BINS, FILES_PER_SCENE, DISP_BATCH_BINS*STR_BATCH_BINS*FILES_PER_SCENE)))
ex_data.assignBatchBins(disp_bins = DISP_BATCH_BINS,         # Number of batch disparity bins
                        str_bins =  STR_BATCH_BINS,          # Number of batch strength bins
                        files_per_scene = FILES_PER_SCENE,   # not used here, will be used when generating batches
                        min_batch_choices=MIN_BATCH_CHOICES, # not used here, will be used when generating batches
                        max_batch_files = MAX_BATCH_FILES)   # not used here, will be used when generating batches

#FILES_PER_SCENE
wait_and_show = False
if DEBUG_LEVEL > 0:
    mytitle = "Disparity_Strength histogram"
    fig = plt.figure()
    fig.canvas.set_window_title(mytitle)
    fig.suptitle(mytitle)
    plt.imshow(ex_data.blurred_hist, vmin=0, vmax=.1 * ex_data.blurred_hist.max())#,vmin=-6,vmax=-2) # , vmin=0, vmax=.01)
    plt.colorbar(orientation='horizontal') # location='bottom')
    bb_display = ex_data.hist_to_batch.copy()
    bb_display = ( 1+ (bb_display % 2) + 2 * ((bb_display % 20)//10)) * (ex_data.hist_to_batch > 0) #).astype(float) 

    fig2 = plt.figure()
    fig2.canvas.set_window_title("Batch indices")
    fig2.suptitle("Batch index for each disparity/strength cell")
    plt.imshow(bb_display) #, vmin=0, vmax=.1 * ex_data.blurred_hist.max())#,vmin=-6,vmax=-2) # , vmin=0, vmax=.01)
    wait_and_show = True

print_time("Creating lists of available correlation data files for each scene")
ex_data.getMLList(ex_data.files_train) # train_list)
print_time("Creating lists of tiles to fall into each DS bin for each scene (long run).")
ex_data.makeBatchLists(train_ds = ex_data.train_ds)
print_time("Done with lists of tiles.")


print_time("Importing TensorCrawl")

import tensorflow as tf
import tensorflow.contrib.slim as slim

print_time("TensorCrawl imported")

result_dir = './result/'
checkpoint_dir = './result/'
save_freq = 500

def lrelu(x):
    #return tf.maximum(x*0.2,x)
    return tf.nn.relu(x)

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
        cost2 = 0.0
        cost3 = 0.0    
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
        residual_disp =  tf.subtract(d_gt_slice, target_disparity_batch, name = "residual_disp")
        out_diff =       tf.subtract(disp_slice, residual_disp,          name = "out_diff")
    out_diff2 =          tf.square(out_diff,                             name = "out_diff2")
    out_wdiff2 =         tf.multiply (out_diff2, w_norm,                 name = "out_wdiff2")
    cost1 =              tf.reduce_sum(out_wdiff2,                       name = "cost1")
    if use_confidence:
        cost12 =         tf.add(cost1,  cost2,                           name = "cost12")
        cost123 =        tf.add(cost12, cost3,                           name = "cost123")
        return cost123, disp_slice, d_gt_slice, out_diff,out_diff2, w_norm
    else:
        return cost1, disp_slice, d_gt_slice, out_diff,out_diff2, w_norm

sess = tf.Session()

#seems that float64 can feed float32!
in_tile =   tf.placeholder(tf.float32,[None,9 * 9 * 4 + 1])
gt =        tf.placeholder(tf.float32,[None,2])
target_d =  tf.placeholder(tf.float32,[None])
out =       network(in_tile)


#Try standard loss functions first
G_loss, _disp_slice, _d_gt_slice, _out_diff, _out_diff2, _w_norm = batchLoss(out_batch =         out,        # [batch_size,(1..2)] tf_result
              target_disparity_batch=  target_d,   # [batch_size]        tf placeholder
              gt_ds_batch =            gt,         # [batch_size,2]      tf placeholder
              absolute_disparity =     ABSOLUTE_DISPARITY,
              use_confidence =         USE_CONFIDENCE, # True, 
              lambda_conf_avg =        0.01,
              lambda_conf_pwr =        0.1,
              conf_pwr =               2.0,
              gt_conf_offset =         0.08,
              gt_conf_pwr =            1.0)

t_vars=tf.trainable_variables()
lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

saver=tf.train.Saver()
# ?!!!!!
#merged = tf.summary.merge_all()
#train_writer = tf.summary.FileWriter(result_dir + '/train', sess.graph)
#test_writer = tf.summary.FileWriter(result_dir + '/test')

sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)

if ckpt:
  print('loaded '+ckpt.model_checkpoint_path)
  saver.restore(sess,ckpt.model_checkpoint_path)


allfolders = glob.glob('./result/*0')
lastepoch = 0
for folder in allfolders:
  lastepoch = np.maximum(lastepoch, int(folder[-4:]))

recorded_loss = []
recorded_mean_loss = []

recorded_gt_d = []
recorded_gt_c = []

recorded_pr_d = []
recorded_pr_c = []

LR = 1e-3

print(bcolors.HEADER+"Last Epoch = "+str(lastepoch)+bcolors.ENDC)

if DEBUG_PLT_LOSS:
  plt.ion()   # something about plotting
  plt.figure(1, figsize=(4,12))
  pass


training_tiles  = np.array([])
training_values = np.array([])



graph_saved = False
for epoch in range(20): #MAX_EPOCH):
    print_time("epoch="+str(epoch))
    train_seed_list = np.arange(len(ex_data.files_train))
    np.random.shuffle(train_seed_list)
    g_loss = np.zeros(len(train_seed_list))
    for nscene, seed_index in enumerate(train_seed_list):
        corr2d_batch, target_disparity_batch, gt_ds_batch = ex_data.prepareBatchData(seed_index)
        num_tiles =         corr2d_batch.shape[0] # 1000
        num_tile_slices =   corr2d_batch.shape[1] # 4
        num_cell_in_slice = corr2d_batch.shape[2] # 81
        in_data = np.empty((num_tiles, num_tile_slices*num_cell_in_slice + 1), dtype = np.float32)
        in_data[...,0:num_tile_slices*num_cell_in_slice] = corr2d_batch.reshape((corr2d_batch.shape[0],corr2d_batch.shape[1]*corr2d_batch.shape[2]))
        in_data[...,num_tile_slices*num_cell_in_slice] =  target_disparity_batch
        st=time.time()
        
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        #_,G_current,output = sess.run([G_opt,G_loss,out],feed_dict={in_tile:input_patch,gt:gt_patch,lr:LR},options=run_options,run_metadata=run_metadata)

        print_time("%d:%d Run "%(epoch, nscene), end = "")
        _,G_current,output, disp_slice, d_gt_slice, out_diff, out_diff2, w_norm = sess.run([G_opt,G_loss,out,_disp_slice, _d_gt_slice, _out_diff, _out_diff2, _w_norm],
                                      feed_dict={in_tile:  in_data,
                                                 gt:       gt_ds_batch,
                                                 target_d: target_disparity_batch, 
                                                 lr:       LR})
        if not graph_saved:
            writer = tf.summary.FileWriter('./attic/nn_ds_single_graph1', sess.graph)
            writer.close()
            graph_saved = True
#            exit(0)
        
        g_loss[nscene]=G_current
        mean_loss = np.mean(g_loss[np.where(g_loss)])
        print_time("loss=%f, running average=%f"%(G_current,mean_loss))
        pass

"""

"""






if wait_and_show: # wait and show images
    plt.show()
print_time("All done, exiting...")   