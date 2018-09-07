#!/usr/bin/env python3
##from numpy import float64
##from tensorflow.contrib.losses.python.metric_learning.metric_loss_ops import npairs_loss
##from debian.deb822 import PdiffIndex

__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "andrey@elphel.com"

#python3 nn_ds_neibs17.py /home/eyesis/x3d_data/data_sets/conf/qcstereo_conf13.xml /home/eyesis/x3d_data/data_sets
import os
import sys
import numpy as np
import time
import shutil
from threading import Thread
import qcstereo_network
import qcstereo_losses
import qcstereo_functions as qsf

qsf.TIME_START = time.time()
qsf.TIME_LAST  = qsf.TIME_START

IMG_WIDTH =        324 # tiles per image row
DEBUG_LEVEL= 1

try:
    conf_file =  sys.argv[1]
except IndexError:
    print("Configuration path is required as a first argument. Optional second argument specifies root directory for data files")
    exit(1)
try:
    root_dir =  sys.argv[2]
except IndexError:
    root_dir =  os.path.dirname(conf_file)
    
print ("Configuration file: " + conf_file)
parameters, dirs, files, _ = qsf.parseXmlConfig(conf_file, root_dir)
"""
Temporarily for backward compatibility
"""
if not "SLOSS_CLIP" in parameters:
    parameters['SLOSS_CLIP'] = 0.5
    print ("Old config, setting SLOSS_CLIP=", parameters['SLOSS_CLIP'])
    
"""
Defined in config file
"""
TILE_SIDE, TILE_LAYERS, TWO_TRAINS, NET_ARCH1, NET_ARCH2 = [None]*5
ABSOLUTE_DISPARITY,SYM8_SUB, WLOSS_LAMBDA,  SLOSS_LAMBDA, SLOSS_CLIP  = [None]*5
SPREAD_CONVERGENCE, INTER_CONVERGENCE, HOR_FLIP, DISP_DIFF_CAP, DISP_DIFF_SLOPE  = [None]*5
CLUSTER_RADIUS = None
PARTIALS_WEIGHTS, MAX_IMGS_IN_MEM, MAX_FILES_PER_GROUP,  BATCH_WEIGHTS, ONLY_TILE = [None] * 5  
USE_CONFIDENCE, WBORDERS_ZERO, EPOCHS_TO_RUN, FILE_UPDATE_EPOCHS = [None] * 4
LR600,LR400,LR200,LR100,LR = [None]*5
SHUFFLE_FILES, EPOCHS_FULL_TEST, SAVE_TIFFS = [None] * 3

TRAIN_BUFFER_GPU, TRAIN_BUFFER_CPU = [None]*2

"""
Next gets globals from the config file
"""
globals().update(parameters)


TRAIN_BUFFER_SIZE = TRAIN_BUFFER_GPU * TRAIN_BUFFER_CPU # in merged (quad) batches



#exit(0)

TILE_SIZE =         TILE_SIDE* TILE_SIDE # == 81
FEATURES_PER_TILE =  TILE_LAYERS * TILE_SIZE# == 324
BATCH_SIZE =       ([1,2][TWO_TRAINS])*2*1000//25 # == 80 Each batch of tiles has balanced D/S tiles, shuffled batches but not inside batches

SUFFIX=(str(NET_ARCH1)+'-'+str(NET_ARCH2)+
       (["R","A"][ABSOLUTE_DISPARITY]) +
       (["NS","S8"][SYM8_SUB])+
       "WLAM"+str(WLOSS_LAMBDA)+
       "SLAM"+str(SLOSS_LAMBDA)+
       "SCLP"+str(SLOSS_CLIP)+
       (['_nG','_G'][SPREAD_CONVERGENCE])+
       (['_nI','_I'][INTER_CONVERGENCE]) +
       (['_nHF',"_HF"][HOR_FLIP]) +
       ('_CP'+str(DISP_DIFF_CAP)) +
       ('_S'+str(DISP_DIFF_SLOPE))
       )
NN_LAYOUTS = {0:[0,   0,   0,   32,  20,  16],
              1:[0,   0,   0,  256, 128,  64],
              2:[0, 128,  32,   32,  32,  16],
              3:[0,   0,  40,   32,  20,  16],
              4:[0,   0,   0,    0,  16,  16],
              5:[0,   0,  64,   32,  32,  16],
              6:[0,   0,  32,   16,  16,  16],
              7:[0,   0,  64,   16,  16,  16],
              8:[0,   0,   0,   64,  20,  16],
              9:[0,   0, 256,   64,  32,  16],
             10:[0, 256, 128,   64,  32,  16],
             11:[0,   0,   0,   0,   64,  32],
              }
NN_LAYOUT1 = NN_LAYOUTS[NET_ARCH1]
NN_LAYOUT2 = NN_LAYOUTS[NET_ARCH2]
USE_PARTIALS =      not PARTIALS_WEIGHTS is None # False - just a single Siamese net, True - partial outputs that use concentric squares of the first level subnets
##############################################################################
cluster_size = (2 * CLUSTER_RADIUS + 1) * (2 * CLUSTER_RADIUS + 1)
center_tile_index = 2 * CLUSTER_RADIUS * (CLUSTER_RADIUS + 1)
qsf.prepareFiles(dirs, files, suffix = SUFFIX)

partials = None
partials = qsf.concentricSquares(CLUSTER_RADIUS)
PARTIALS_WEIGHTS = [1.0*pw/sum(PARTIALS_WEIGHTS) for pw in PARTIALS_WEIGHTS]
if not USE_PARTIALS:
    partials = partials[0:1]
    PARTIALS_WEIGHTS = [1.0]


import tensorflow as tf
#import tensorflow.contrib.slim as slim

qsf.evaluateAllResults(result_files = files['result'],
                       absolute_disparity = ABSOLUTE_DISPARITY,
                       cluster_radius = CLUSTER_RADIUS)

image_data = qsf.initImageData(
                files =          files,
                max_imgs =       MAX_IMGS_IN_MEM,
                cluster_radius = CLUSTER_RADIUS,
                tile_layers =    TILE_LAYERS,
                tile_side =      TILE_SIDE,
                width =          IMG_WIDTH,
                replace_nans =   True)
    
#    return train_next, dataset_train_all, datasets_test
corr2d_len, target_disparity_len, _ = qsf.get_lengths(CLUSTER_RADIUS, TILE_LAYERS, TILE_SIDE)
 
train_next, dataset_train, datasets_test= qsf.initTrainTestData(
        files = files,
        cluster_radius =      CLUSTER_RADIUS,
        buffer_size =         TRAIN_BUFFER_SIZE * BATCH_SIZE) # number of clusters per train
##    return corr2d_len, target_disparity_len, train_next, dataset_train_merged, datasets_test

    
corr2d_train_placeholder =           tf.placeholder(dataset_train.dtype, (None,FEATURES_PER_TILE * cluster_size)) # corr2d_train.shape)
target_disparity_train_placeholder = tf.placeholder(dataset_train.dtype, (None,1 *   cluster_size))  #target_disparity_train.shape)
gt_ds_train_placeholder =            tf.placeholder(dataset_train.dtype, (None,2 *   cluster_size)) #gt_ds_train.shape)

dataset_tt = tf.data.Dataset.from_tensor_slices({
    "corr2d":           corr2d_train_placeholder,
    "target_disparity": target_disparity_train_placeholder,
    "gt_ds":            gt_ds_train_placeholder})

tf_batch_weights = tf.placeholder(shape=(None,), dtype=tf.float32, name = "batch_weights") # way to increase importance of the high variance clusters 
feed_batch_weights =   np.array(BATCH_WEIGHTS*(BATCH_SIZE//len(BATCH_WEIGHTS)), dtype=np.float32)
feed_batch_weight_1 =  np.array([1.0], dtype=np.float32) 

##dataset_train_size = len(datasets_train[0]['corr2d'])
##dataset_train_size //= BATCH_SIZE

#dataset_train_size = TRAIN_BUFFER_GPU * num_train_subs # TRAIN_BUFFER_SIZE

#dataset_test_size = len(datasets_test[0]['corr2d'])
dataset_test_size = len(datasets_test[0])
dataset_test_size //= BATCH_SIZE
#dataset_img_size = len(datasets_img[0]['corr2d'])
dataset_img_size = len(image_data[0]['corr2d'])
dataset_img_size //= BATCH_SIZE

dataset_tt = dataset_tt.batch(BATCH_SIZE)
dataset_tt = dataset_tt.prefetch(BATCH_SIZE)
iterator_tt = dataset_tt.make_initializable_iterator()
next_element_tt = iterator_tt.get_next()

#https://www.tensorflow.org/versions/r1.5/programmers_guide/datasets
result_dir = './attic/result_neibs_'+     SUFFIX+'/'
checkpoint_dir = './attic/result_neibs_'+ SUFFIX+'/'
save_freq = 500

def debug_gt_variance(
        indx,        # This tile index (0..8)
        center_indx, # center tile index
        gt_ds_batch # [?:9:2]
        ):
    with tf.name_scope("Debug_GT_Variance"):
#        tf_num_tiles =  tf.shape(gt_ds_batch)[0]
        d_gt_this =     tf.reshape(gt_ds_batch[:,2 * indx],[-1],                     name = "d_this")
        d_gt_center =   tf.reshape(gt_ds_batch[:,2 * center_indx],[-1],              name = "d_center")
        d_gt_diff =     tf.subtract(d_gt_this, d_gt_center,                          name = "d_diff")
        d_gt_diff2 =    tf.multiply(d_gt_diff, d_gt_diff,                            name = "d_diff2")
        d_gt_var =      tf.reduce_mean(d_gt_diff2,                                   name = "d_gt_var")
        return  d_gt_var
    
#def batchLoss
        
        
target_disparity_cluster = tf.reshape(next_element_tt['target_disparity'], [-1,cluster_size, 1], name="targdisp_cluster")    
corr2d_Nx325 = tf.concat([tf.reshape(next_element_tt['corr2d'],[-1,cluster_size,FEATURES_PER_TILE], name="coor2d_cluster"),
                          target_disparity_cluster], axis=2, name = "corr2d_Nx325")
if SPREAD_CONVERGENCE:                                      
    outs, inp_weights =  qcstereo_network.networks_siam(
                                            input =             corr2d_Nx325,
                                            input_global =      target_disparity_cluster,
                                            layout1 =           NN_LAYOUT1, 
                                            layout2 =           NN_LAYOUT2,
                                            inter_convergence = INTER_CONVERGENCE,
                                            sym8 =              SYM8_SUB,
                                            only_tile =         ONLY_TILE, #Remove/put None for normal operation
                                            partials =          partials,
                                            use_confidence=     USE_CONFIDENCE)
                                            
else:
    outs, inp_weights =  qcstereo_network.networks_siam(
                                            input_tensor=       corr2d_Nx325,
                                            input_global =      None,
                                            layout1 =           NN_LAYOUT1, 
                                            layout2 =           NN_LAYOUT2,
                                            inter_convergence = False,
                                            sym8 =              SYM8_SUB,
                                            only_tile =         ONLY_TILE, #Remove/put None for normal operation
                                            partials =          partials,
                                            use_confidence=     USE_CONFIDENCE)
                                                                                      
tf_partial_weights = tf.constant(PARTIALS_WEIGHTS,dtype=tf.float32,name="partial_weights")
G_losses = [0.0]*len(partials)
target_disparity_batch=  next_element_tt['target_disparity'][:,center_tile_index:center_tile_index+1]
gt_ds_batch_clust =      next_element_tt['gt_ds']
#gt_ds_batch =            next_element_tt['gt_ds'][:,2 * center_tile_index: 2 * (center_tile_index +1)]
gt_ds_batch =            gt_ds_batch_clust[:,2 * center_tile_index: 2 * (center_tile_index +1)]
G_losses[0], _disp_slice, _d_gt_slice, _out_diff, _out_diff2, _w_norm, _out_wdiff2, _cost1 = qcstereo_losses.batchLoss(
              out_batch =              outs[0],        # [batch_size,(1..2)] tf_result
              target_disparity_batch=  target_disparity_batch, # next_element_tt['target_disparity'][:,center_tile_index:center_tile_index+1], # target_disparity_batch_center, # next_element_tt['target_disparity'], # target_disparity, ### target_d,   # [batch_size]        tf placeholder
              gt_ds_batch =            gt_ds_batch, # next_element_tt['gt_ds'][:,2 * center_tile_index: 2 * (center_tile_index +1)],  # gt_ds_batch_center, ## next_element_tt['gt_ds'], # gt_ds, ### gt,         # [batch_size,2]      tf placeholder
              batch_weights =          tf_batch_weights,
              disp_diff_cap =          DISP_DIFF_CAP,
              disp_diff_slope=         DISP_DIFF_SLOPE,
              absolute_disparity =     ABSOLUTE_DISPARITY,
              use_confidence =         USE_CONFIDENCE, # True, 
              lambda_conf_avg =        0.01,
##              lambda_conf_pwr =        0.1,
              conf_pwr =               2.0,
              gt_conf_offset =         0.08,
              gt_conf_pwr =            2.0,
              error2_offset =          0, # 0.0025, # (0.05^2)
              disp_wmin =              1.0,    # minimal disparity to apply weight boosting for small disparities
              disp_wmax =              8.0,    # maximal disparity to apply weight boosting for small disparities
              use_out =                False)  # use calculated disparity for disparity weight boosting (False - use target disparity)

G_loss = G_losses[0]
for n in range (1,len(partials)):
    G_losses[n], _, _, _, _, _, _, _ = qcstereo_losses.batchLoss(
              out_batch =              outs[n],        # [batch_size,(1..2)] tf_result
              target_disparity_batch=  target_disparity_batch, #next_element_tt['target_disparity'][:,center_tile_index:center_tile_index+1], # target_disparity_batch_center, # next_element_tt['target_disparity'], # target_disparity, ### target_d,   # [batch_size]        tf placeholder
              gt_ds_batch =            gt_ds_batch, # next_element_tt['gt_ds'][:,2 * center_tile_index: 2 * (center_tile_index +1)],  # gt_ds_batch_center, ## next_element_tt['gt_ds'], # gt_ds, ### gt,         # [batch_size,2]      tf placeholder
              batch_weights =          tf_batch_weights,
              disp_diff_cap =          DISP_DIFF_CAP,
              disp_diff_slope=         DISP_DIFF_SLOPE,
              absolute_disparity =     ABSOLUTE_DISPARITY,
              use_confidence =         USE_CONFIDENCE, # True, 
              lambda_conf_avg =        0.01,
#              lambda_conf_pwr =        0.1,
              conf_pwr =               2.0,
              gt_conf_offset =         0.08,
              gt_conf_pwr =            2.0,
              error2_offset =          0, # 0.0025, # (0.05^2)
              disp_wmin =              1.0,    # minimal disparity to apply weight boosting for small disparities
              disp_wmax =              8.0,    # maximal disparity to apply weight boosting for small disparities
              use_out =                False)  # use calculated disparity for disparity weight boosting (False - use target disparity)

tf_wlosses = tf.multiply(G_losses, tf_partial_weights, name =  "tf_wlosses")
G_losses_sum = tf.reduce_sum(tf_wlosses, name = "G_losses_sum")

if SLOSS_LAMBDA > 0:    
    S_loss, rslt_cost_nw, rslt_cost_w, rslt_d , rslt_avg_disparity, rslt_gt_disparity, rslt_offs = qcstereo_losses.smoothLoss(
               out_batch =             outs[0],                   # [batch_size,(1..2)] tf_result
               target_disparity_batch = target_disparity_batch,    # [batch_size]        tf placeholder
               gt_ds_batch_clust =      gt_ds_batch_clust,           # [batch_size,25,2]      tf placeholder
               clip =                   SLOSS_CLIP,
               absolute_disparity =     ABSOLUTE_DISPARITY, #when false there should be no activation on disparity output !
               cluster_radius =         CLUSTER_RADIUS)
    GS_loss =  tf.add(G_losses_sum, SLOSS_LAMBDA * S_loss, name = "GS_loss")

else:
    S_loss =   tf.constant(0.0, dtype=tf.float32,name = "S_loss")
    GS_loss = G_losses_sum # G_loss
        
#    G_loss +=  Glosses[n]*PARTIALS_WEIGHTS[n]
#tf_partial_weights
if WLOSS_LAMBDA > 0.0:   
    W_loss =     qcstereo_losses.weightsLoss(
        inp_weights =   inp_weights[0], #    inp_weights - list of tensors, currently - just [0]
        tile_layers=    TILE_LAYERS, # 4
        tile_side =     TILE_SIDE, # 9
        wborders_zero = WBORDERS_ZERO)

#    GW_loss =    tf.add(G_loss, WLOSS_LAMBDA * W_loss, name = "GW_loss")
    GW_loss =    tf.add(GS_loss, WLOSS_LAMBDA * W_loss, name = "GW_loss")
else:
    GW_loss =    GS_loss # G_loss
    W_loss =     tf.constant(0.0, dtype=tf.float32,name = "W_loss")
#debug
GT_variance =  debug_gt_variance(indx = 0,        # This tile index (0..8)
                                 center_indx = 4, # center tile index
                                 gt_ds_batch = next_element_tt['gt_ds'])# [?:18]
              
tf_ph_G_loss =    tf.placeholder(tf.float32,shape=None,name='G_loss_avg')
tf_ph_G_losses =  tf.placeholder(tf.float32,shape=[len(partials)],name='G_losses_avg')
tf_ph_S_loss =    tf.placeholder(tf.float32,shape=None,name='S_loss_avg')
tf_ph_W_loss =    tf.placeholder(tf.float32,shape=None,name='W_loss_avg')
tf_ph_GW_loss =   tf.placeholder(tf.float32,shape=None,name='GW_loss_avg')
tf_ph_sq_diff =   tf.placeholder(tf.float32,shape=None,name='sq_diff_avg')
tf_gtvar_diff =   tf.placeholder(tf.float32,shape=None,name='gtvar_diff')
tf_img_test0 =    tf.placeholder(tf.float32,shape=None,name='img_test0')
tf_img_test9 =    tf.placeholder(tf.float32,shape=None,name='img_test9')
with tf.name_scope('sample'):
    tf.summary.scalar("GW_loss",      GW_loss)
    tf.summary.scalar("G_loss",       G_loss)
    tf.summary.scalar("S_loss",       S_loss)
    tf.summary.scalar("W_loss",       W_loss)
    tf.summary.scalar("sq_diff",      _cost1)
    tf.summary.scalar("gtvar_diff",   GT_variance)
    
with tf.name_scope('epoch_average'):
#    for i, tl in enumerate(tf_ph_G_losses):
#       tf.summary.scalar("GW_loss_epoch_"+str(i), tl)
    for i in range(tf_ph_G_losses.shape[0]):
        tf.summary.scalar("G_loss_epoch_"+str(i), tf_ph_G_losses[i])
        
    tf.summary.scalar("GW_loss_epoch", tf_ph_GW_loss)
    tf.summary.scalar("G_loss_epoch",  tf_ph_G_loss)
    tf.summary.scalar("S_loss_epoch",  tf_ph_S_loss)
    tf.summary.scalar("W_loss_epoch",  tf_ph_W_loss)
    tf.summary.scalar("sq_diff_epoch", tf_ph_sq_diff)
    tf.summary.scalar("gtvar_diff",    tf_gtvar_diff)
    
    tf.summary.scalar("img_test0",     tf_img_test0)
    tf.summary.scalar("img_test9",     tf_img_test9)

t_vars=            tf.trainable_variables()
lr=                tf.placeholder(tf.float32)
G_opt=             tf.train.AdamOptimizer(learning_rate=lr).minimize(GW_loss)


ROOT_PATH  = './attic/nn_ds_neibs17_graph'+SUFFIX+"/"
TRAIN_PATH =  ROOT_PATH + 'train'
TEST_PATH  =  ROOT_PATH + 'test'
TEST_PATH1  = ROOT_PATH + 'test1'

# CLEAN OLD STAFF
shutil.rmtree(TRAIN_PATH, ignore_errors=True)
shutil.rmtree(TEST_PATH, ignore_errors=True)
shutil.rmtree(TEST_PATH1, ignore_errors=True)

WIDTH=324
HEIGHT=242

num_train_subs = len(train_next) # number of (different type) merged training sets    
dataset_train_size = TRAIN_BUFFER_GPU * num_train_subs # TRAIN_BUFFER_SIZE

with tf.Session()  as sess:
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    merged = tf.summary.merge_all()
    train_writer =    tf.summary.FileWriter(TRAIN_PATH, sess.graph)
    test_writer  =    tf.summary.FileWriter(TEST_PATH, sess.graph)
    test_writer1  =   tf.summary.FileWriter(TEST_PATH1, sess.graph)
    
    loss_gw_train_hist=  np.empty(dataset_train_size, dtype=np.float32)
    loss_g_train_hist=   np.empty(dataset_train_size, dtype=np.float32)
    
    loss_g_train_hists=   [np.empty(dataset_train_size, dtype=np.float32) for p in partials]
    
    
    loss_s_train_hist=   np.empty(dataset_train_size, dtype=np.float32)
    loss_w_train_hist=   np.empty(dataset_train_size, dtype=np.float32)
    
    loss_gw_test_hist=  np.empty(dataset_test_size, dtype=np.float32)
#    loss_g_test_hist=   np.empty(dataset_test_size, dtype=np.float32)
    loss_g_test_hists=   [np.empty(dataset_test_size, dtype=np.float32) for p in partials]
    
    loss_s_test_hist=   np.empty(dataset_test_size, dtype=np.float32)
    loss_w_test_hist=   np.empty(dataset_test_size, dtype=np.float32)
    
    loss2_train_hist= np.empty(dataset_train_size, dtype=np.float32)
    loss2_test_hist=  np.empty(dataset_test_size, dtype=np.float32)
    train_gw_avg = 0.0
    train_g_avg =  0.0
    train_g_avgs =  [0.0]*len(partials)
    
    train_w_avg =  0.0
    train_s_avg =  0.0
    test_gw_avg =  0.0     
    test_g_avg =   0.0     
    test_g_avgs =  [0.0]*len(partials)
    test_w_avg =   0.0     
    test_s_avg =   0.0     

    train2_avg = 0.0
    test2_avg = 0.0
    gtvar_train_hist=  np.empty(dataset_train_size, dtype=np.float32)
    gtvar_test_hist=   np.empty(dataset_test_size, dtype=np.float32)
    gtvar_train = 0.0
    gtvar_test = 0.0
    gtvar_train_avg = 0.0
    gtvar_test_avg =  0.0
    img_gain_test0 =  1.0
    img_gain_test9 =  1.0
    
    thr=None
    thr_result = None
    trains_to_update = [train_next[n_train]['more_files'] for n_train in range(len(train_next))]
    for epoch in range (EPOCHS_TO_RUN):
        """
        update files after each epoch, all 4.
        Convert to threads after testing
        """
        if (FILE_UPDATE_EPOCHS > 0) and (epoch % FILE_UPDATE_EPOCHS == 0):
            if not thr is None:
                if thr.is_alive():
                    qsf.print_time("***WAITING*** until tfrecord gets loaded", end=" ")
                else:
                    qsf.print_time("tfrecord is ***ALREADY LOADED*** loaded", end=" ")
        
                thr.join()
                qsf.print_time("Done")
                qsf.print_time("Inserting new data", end=" ")
                for n_train in range(len(trains_to_update)):
                    if trains_to_update[n_train]:
                        qsf.add_file_to_dataset(dataset = dataset_train,
                                                new_dataset = thr_result[n_train],
                                                train_next = train_next[n_train])
                qsf.print_time("Done")
            thr_result = []
            fpaths = []
            for n_train in range(len(trains_to_update)):
                if trains_to_update[n_train]:
                    fpaths.append(files['train'][n_train][train_next[n_train]['file']])
                    qsf.print_time("Will read in background: "+fpaths[-1])
            thr = Thread(target=qsf.getMoreFiles, args=(fpaths,thr_result, CLUSTER_RADIUS, HOR_FLIP, TILE_LAYERS, TILE_SIDE))            
            thr.start()        
        train_buf_index = epoch %   TRAIN_BUFFER_CPU # GPU memory from CPU memory (now 4)
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
        if (train_buf_index == 0) and SHUFFLE_FILES:
            qsf.print_time("Shuffling how datasets datasets_train_lvar and datasets_train_hvar are zipped together", end="")
            qsf.shuffle_in_place(
                dataset_data = dataset_train, #alternating clusters from 4 sources.each cluster has all needed data (concatenated)
                period = num_train_subs)
            qsf.print_time("  Done")
        sti = train_buf_index *  dataset_train_size * BATCH_SIZE #      TRAIN_BUFFER_GPU * num_train_subs
        eti = sti+   dataset_train_size * BATCH_SIZE#    (train_buf_index +1) *  TRAIN_BUFFER_GPU * num_train_subs
         
        sess.run(iterator_tt.initializer, feed_dict={corr2d_train_placeholder:           dataset_train[sti:eti,:corr2d_len], 
                                                     target_disparity_train_placeholder: dataset_train[sti:eti,corr2d_len:corr2d_len+target_disparity_len],
                                                     gt_ds_train_placeholder:            dataset_train[sti:eti,corr2d_len+target_disparity_len:] })
        
        
        for i in range(dataset_train_size):
            try:
#                train_summary,_, GW_loss_trained,  G_loss_trained,  W_loss_trained,  output, disp_slice, d_gt_slice, out_diff, out_diff2, w_norm, out_wdiff2, out_cost1, gt_variance  = sess.run(
                train_summary,_, GW_loss_trained,  G_losses_trained,  S_loss_trained,  W_loss_trained,  output, disp_slice, d_gt_slice, out_diff, out_diff2, w_norm, out_wdiff2, out_cost1, gt_variance  = sess.run(
                    [   merged,
                        G_opt,
                        GW_loss,
#                        G_loss,
                        G_losses,
                        S_loss,
                        W_loss,
                        outs[0],
                        _disp_slice,
                        _d_gt_slice,
                        _out_diff,
                        _out_diff2,
                        _w_norm,
                        _out_wdiff2,
                        _cost1,
                        GT_variance
                    ],
                    feed_dict={tf_batch_weights: feed_batch_weights,
                               lr:               learning_rate,
                               tf_ph_GW_loss:    train_gw_avg,
                               tf_ph_G_loss:     train_g_avgs[0], #train_g_avg,
                               tf_ph_G_losses:   train_g_avgs,
                               tf_ph_S_loss:     train_s_avg,
                               tf_ph_W_loss:     train_w_avg,
                               tf_ph_sq_diff:    train2_avg,
                               tf_gtvar_diff:    gtvar_train_avg,
                               tf_img_test0:     img_gain_test0,
                               tf_img_test9:     img_gain_test9}) # previous value of *_avg #Fetch argument 0.0 has invalid type <class 'float'>, must be a string or Tensor. (Can not convert a float into a Tensor or Operation.)
                
                loss_gw_train_hist[i] = GW_loss_trained
                for nn, gl  in enumerate(G_losses_trained):
                    loss_g_train_hists[nn][i] =  gl
                loss_s_train_hist[i] =  S_loss_trained
                loss_w_train_hist[i] =  W_loss_trained
                loss2_train_hist[i] = out_cost1
                gtvar_train_hist[i] = gt_variance
            except tf.errors.OutOfRangeError:
                print("****** NO MORE DATA! train done at step %d"%(i))
                break
#            print ("==== i=%d, GW_loss_trained=%f  loss_gw_train_hist[%d]=%f ===="%(i,GW_loss_trained,i,loss_gw_train_hist[i]))

        train_gw_avg =      np.average(loss_gw_train_hist).astype(np.float32)     
        train_g_avg =       np.average(loss_g_train_hist).astype(np.float32) 
        for nn, lgth  in enumerate(loss_g_train_hists):
            train_g_avgs[nn] =       np.average(lgth).astype(np.float32)
###############        
        train_s_avg =       np.average(loss_s_train_hist).astype(np.float32)     
        train_w_avg =       np.average(loss_w_train_hist).astype(np.float32)     
        train2_avg =      np.average(loss2_train_hist).astype(np.float32)
        gtvar_train_avg = np.average(gtvar_train_hist).astype(np.float32)
        
        test_summaries = [0.0]*len(datasets_test)
        tst_avg =        [0.0]*len(datasets_test)
        tst2_avg =       [0.0]*len(datasets_test)
        for ntest,dataset_test in enumerate(datasets_test):
            sess.run(iterator_tt.initializer, feed_dict={corr2d_train_placeholder:      dataset_test[:, :corr2d_len],  #['corr2d'],
                                                    target_disparity_train_placeholder: dataset_test[:, corr2d_len:corr2d_len+target_disparity_len], # ['target_disparity'],
                                                    gt_ds_train_placeholder:            dataset_test[:, corr2d_len+target_disparity_len:] }) # ['gt_ds']})
            
            for i in range(dataset_test_size):
                try:
                    test_summaries[ntest], GW_loss_tested, G_losses_tested, S_loss_tested, W_loss_tested, output, disp_slice, d_gt_slice, out_diff, out_diff2, w_norm, out_wdiff2, out_cost1, gt_variance = sess.run(
                        [merged,
                         GW_loss,
                         G_losses,
                         S_loss,
                         W_loss,
                         outs[0],
                         _disp_slice,
                         _d_gt_slice,
                         _out_diff,
                         _out_diff2,
                         _w_norm,
                         _out_wdiff2,
                         _cost1,
                         GT_variance
                         ],
                         feed_dict={tf_batch_weights: feed_batch_weight_1 , #  feed_batch_weights,
                                    lr:               learning_rate,
                                    tf_ph_GW_loss:    test_gw_avg,
                                    tf_ph_G_loss:     test_g_avg,
                                    tf_ph_G_losses:   test_g_avgs, # train_g_avgs, # temporary, there is o data fro test
                                    tf_ph_S_loss:     test_s_avg,
                                    tf_ph_W_loss:     test_w_avg,
                                    tf_ph_sq_diff:    test2_avg,
                                    tf_gtvar_diff:    gtvar_test_avg,
                                    tf_img_test0:     img_gain_test0,
                                    tf_img_test9:     img_gain_test9})  # previous value of *_avg
                    loss_gw_test_hist[i] =  GW_loss_tested
                    
                    for nn, gl  in enumerate(G_losses_tested):
                        loss_g_test_hists[nn][i] =  gl

                    loss_s_test_hist[i] =   S_loss_tested
                    loss_w_test_hist[i] =   W_loss_tested
                    loss2_test_hist[i] = out_cost1
                    gtvar_test_hist[i] = gt_variance
                except tf.errors.OutOfRangeError:
                    print("test done at step %d"%(i))
                    break
                    
            test_gw_avg =  np.average(loss_gw_test_hist).astype(np.float32)
            
            for nn, lgth  in enumerate(loss_g_test_hists):
                test_g_avgs[nn] =       np.average(lgth).astype(np.float32)
            
            test_s_avg =  np.average(loss_s_test_hist).astype(np.float32)
            test_w_avg =  np.average(loss_w_test_hist).astype(np.float32)
            tst_avg[ntest] =  test_gw_avg   
            test2_avg = np.average(loss2_test_hist).astype(np.float32)
            tst2_avg[ntest] =  test2_avg   
            gtvar_test_avg = np.average(gtvar_test_hist).astype(np.float32)
             
        train_writer.add_summary(train_summary, epoch)
        test_writer.add_summary(test_summaries[0], epoch)
        test_writer1.add_summary(test_summaries[1], epoch)
        
        qsf.print_time("==== %d:%d -> %f %f %f (%f %f %f) dbg:%f %f ===="%(epoch,i,train_gw_avg, tst_avg[0], tst_avg[1], train2_avg, tst2_avg[0], tst2_avg[1], gtvar_train_avg, gtvar_test_avg))
        if (((epoch + 1) == EPOCHS_TO_RUN) or (((epoch + 1) % EPOCHS_FULL_TEST) == 0)) and (len(image_data) > 0) :
            if (epoch + 1) == EPOCHS_TO_RUN: # last
                print("Last epoch, removing train/test datasets to reduce memory footprint")
                del(dataset_train)
                del(dataset_test)             
            last_epoch = (epoch + 1) == EPOCHS_TO_RUN
            ind_img = [0]
            if last_epoch:
                ind_img = [i for i in range(len(image_data))]
###################################################
# Read the full image
################################################### 
            test_summaries_img = [0.0]*len(ind_img) # datasets_img)
            disp_out=     np.empty((WIDTH*HEIGHT), dtype=np.float32)
            dbg_cost_nw=  np.empty((WIDTH*HEIGHT), dtype=np.float32)
            dbg_cost_w=   np.empty((WIDTH*HEIGHT), dtype=np.float32)
            dbg_d=        np.empty((WIDTH*HEIGHT), dtype=np.float32)
            
            dbg_avg_disparity = np.empty((WIDTH*HEIGHT), dtype=np.float32)
            dbg_gt_disparity =  np.empty((WIDTH*HEIGHT), dtype=np.float32)
            dbg_offs =          np.empty((WIDTH*HEIGHT), dtype=np.float32)
            
            for ntest in ind_img: # datasets_img):
                dataset_img = qsf.readImageData(
                    image_data =     image_data,
                    files =          files,
                    indx =           ntest,
                    cluster_radius = CLUSTER_RADIUS,
                    tile_layers =    TILE_LAYERS,
                    tile_side =      TILE_SIDE,
                    width =          IMG_WIDTH,
                    replace_nans =   True)

                sess.run(iterator_tt.initializer, feed_dict={corr2d_train_placeholder:      dataset_img['corr2d'],
                                                        target_disparity_train_placeholder: dataset_img['target_disparity'],
                                                        gt_ds_train_placeholder:            dataset_img['gt_ds']})
                for start_offs in range(0,disp_out.shape[0],BATCH_SIZE):
                    end_offs = min(start_offs+BATCH_SIZE,disp_out.shape[0])
                    
                    try:
                        test_summaries_img[ntest],output, cost_nw, cost_w, dd, avg_disparity, gt_disparity, offs  = sess.run(
                            [merged,
                             outs[0],     # {?,1]
                             rslt_cost_nw, #[?,]
                             rslt_cost_w,  #[?,]
                             rslt_d,       #[?,]
                             
                             rslt_avg_disparity,
                             rslt_gt_disparity,
                             rslt_offs                       
                             ],
                             feed_dict={
                                        tf_batch_weights: feed_batch_weight_1, # feed_batch_weights,
#                                        lr:               learning_rate,
                                        tf_ph_GW_loss:    test_gw_avg,
                                        tf_ph_G_loss:     test_g_avg,
                                        tf_ph_G_losses:   train_g_avgs, # temporary, there is o data for test
                                        tf_ph_S_loss:     test_s_avg,
                                        tf_ph_W_loss:     test_w_avg,
                                        tf_ph_sq_diff:    test2_avg,
                                        tf_gtvar_diff:    gtvar_test_avg,
                                        tf_img_test0:     img_gain_test0,
                                        tf_img_test9:     img_gain_test9})  # previous value of *_avg
                    except tf.errors.OutOfRangeError:
                        print("test done at step %d"%(i))
                        break
                    try:
                        disp_out[start_offs:end_offs] = output.flatten()
                        dbg_cost_nw[start_offs:end_offs] = cost_nw.flatten()
                        dbg_cost_w [start_offs:end_offs] = cost_w.flatten()
                        dbg_d[start_offs:end_offs] = dd.flatten()
                        dbg_avg_disparity[start_offs:end_offs] = avg_disparity.flatten()
                        dbg_gt_disparity[start_offs:end_offs] = gt_disparity.flatten()
                        dbg_offs[start_offs:end_offs] = offs.flatten()
                        
                        
                    except ValueError:
                        print("dataset_img_size= %d, i=%d, output.shape[0]=%d "%(dataset_img_size, i, output.shape[0]))
                        break;    
                    pass
                result_file = files['result'][ntest] # result_files[ntest]
                try:
                    os.makedirs(os.path.dirname(result_file))
                except:
                    pass     

#                rslt = np.concatenate([disp_out.reshape(-1,1), t_disp, gtruth],1)
                rslt = np.concatenate(
                    [disp_out.reshape(-1,1),
                     dataset_img['t_disps'], #t_disps[ntest],
                     dataset_img['gtruths'], # gtruths[ntest],
                     dbg_cost_nw.reshape(-1,1),
                     dbg_cost_w.reshape(-1,1),
                     dbg_d.reshape(-1,1),
                     dbg_avg_disparity.reshape(-1,1),
                     dbg_gt_disparity.reshape(-1,1),
                     dbg_offs.reshape(-1,1)],1)
                np.save(result_file,           rslt.reshape(HEIGHT,WIDTH,-1))
                rslt = qsf.eval_results(result_file, ABSOLUTE_DISPARITY,radius=CLUSTER_RADIUS)                
                img_gain_test0 = rslt[0][0]/rslt[0][1]   
                img_gain_test9 = rslt[9][0]/rslt[9][1]   
                if SAVE_TIFFS:
                    qsf.result_npy_to_tiff(result_file, ABSOLUTE_DISPARITY, fix_nan = True)
                    
                """
                Remove dataset_img (if it is not [0] to reduce memory footprint         
                """
                if ntest > 0:
                    image_data[ntest] = None
     
    # Close writers
    train_writer.close()
    test_writer.close()
    test_writer1.close()
#reports error: Exception ignored in: <bound method BaseSession.__del__ of <tensorflow.python.client.session.Session object at 0x7efc5f720ef0>> if there is no print before exit()

print("All done")
exit (0)
