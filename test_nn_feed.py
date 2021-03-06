#!/usr/bin/env python3

__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "oleg@elphel.com"

'''
Open all tiffs in a folder, combine a single tiff from randomly selected
tiles from originals
'''

from PIL import Image

import os
import sys
import glob

import imagej_tiff as ijt
import pack_tile as pile

import numpy as np
import itertools

import time

import matplotlib.pyplot as plt

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


def print_time():
  print(bcolors.BOLDWHITE+"time: "+str(time.time())+bcolors.ENDC)

# USAGE: python3 test_3.py some-path

VALUES_LAYER_NAME = 'other'
LAYERS_OF_INTEREST = ['diagm-pair', 'diago-pair', 'hor-pairs', 'vert-pairs']
RADIUS = 1
TILE_PACKING_TYPE = 1

DEBUG_PLT_LOSS = True
# If false - will not pack or rescal
DEBUG_PACK_TILES = True

try:
  src = sys.argv[1]
except IndexError:
  src = "."

print("Importing TensorCrawl")
print_time()

import tensorflow as tf
import tensorflow.contrib.slim as slim

print("TensorCrawl imported")
print_time()

tlist = glob.glob(src+"/*.tiff")
print("Found "+str(len(tlist))+" preprocessed tiff files:")
print("\n".join(tlist))
print_time()

tiff = ijt.imagej_tiff(tlist[0])

# get labels
labels = tiff.labels.copy()
labels.remove(VALUES_LAYER_NAME)

print("Image data layers:  "+str(labels))
print("Layers of interest: "+str(LAYERS_OF_INTEREST))
print("Values layer: "+str([VALUES_LAYER_NAME]))


result_dir = './result/'
checkpoint_dir = './result/'
save_freq = 500

def lrelu(x):
    #return tf.maximum(x*0.2,x)
    return tf.nn.relu(x)

def network(input):

  fc1  = slim.fully_connected(input,1024,activation_fn=lrelu,scope='g_fc1')
  fc2  = slim.fully_connected(fc1,     2,activation_fn=lrelu,scope='g_fc2')
  return fc2

  #fc2  = slim.fully_connected(fc1,  1024,activation_fn=lrelu,scope='g_fc2')
  #fc3  = slim.fully_connected(fc2,   512,activation_fn=lrelu,scope='g_fc3')
  #fc4  = slim.fully_connected(fc3,     8,activation_fn=lrelu,scope='g_fc4')
  #fc5  = slim.fully_connected(fc4,     4,activation_fn=lrelu,scope='g_fc5')
  #fc6  = slim.fully_connected(fc5,     2,activation_fn=lrelu,scope='g_fc6')

  #return fc6


sess = tf.Session()

if   TILE_PACKING_TYPE==1:
  in_tile = tf.placeholder(tf.float32,[None,101])
elif TILE_PACKING_TYPE==2:
  in_tile = tf.placeholder(tf.float32,[None,105])

gt = tf.placeholder(tf.float32,[None,2])


#losses    = tf.get_variable("losses", [None])
#update_operation = tf.assign(losses,tf.concat([losses,G_loss]))
#mean_loss = tf.reduce_mean(losses)

#tf.summary.scalar('gt_value', gt[0])
#tf.summary.scalar('gt_confidence', gt[1])
#tf.summary.scalar('gt_value',gt[0,0])

#cf_cutoff = tf.constant(tf.float32,[None,1])
out = network(in_tile)

#tf.summary.scalar('out_value', out[0,0])
#tf.summary.scalar('out_confidence', out[1])

# min cutoff
cf_cutoff = 0.173303
cf_w = tf.pow(tf.maximum(gt[:,1]-cf_cutoff,0.0),1)
#cf_wsum = tf.reduce_sum(cf_w[~tf.is_nan(cf_w)])
#cf_w_norm = cf_w/cf_wsum
cf_w_norm = tf.nn.softmax(cf_w)

#out_cf = out[:,1]

#G_loss = tf.reduce_mean(tf.abs(tf.nn.softmax(out[:,1])*out[:,0]-cf_w_norm*gt[:,0]))
#G_loss = tf.reduce_mean(tf.squared_difference(out[:,0], gt[:,0]))
G_loss = tf.reduce_mean(tf.abs(out[:,0]-gt[:,0]))
#G_loss = tf.losses.mean_squared_error(gt[:,0],out[:,0],cf_w)

#tf.summary.scalar('loss', G_loss)
#tf.summary.scalar('prediction', out[0,0])
#tf.summary.scalar('ground truth', gt[0,0])

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

# get epoch train data
for i in range(len(tlist)):

  print(bcolors.OKGREEN+"Opening "+tlist[i]+bcolors.ENDC)

  tmp_tiff  = ijt.imagej_tiff(tlist[i])
  tmp_tiles = tmp_tiff.getstack(labels,shape_as_tiles=True)
  tmp_vals  = tmp_tiff.getvalues(label=VALUES_LAYER_NAME)
  # might not need it because going to loop through anyway
  if   TILE_PACKING_TYPE==1:
    packed_tiles = pile.pack(tmp_tiles)
  elif TILE_PACKING_TYPE==2:
    packed_tiles = pile.pack(tmp_tiles,TILE_PACKING_TYPE)

  packed_tiles = np.dstack((packed_tiles,tmp_vals[:,:,0]))

  packed_tiles = np.reshape(packed_tiles,(-1,packed_tiles.shape[-1]))
  values       = np.reshape(tmp_vals[:,:,1:3],(-1,2))

  packed_tiles_filtered = np.array([])

  print("Unfiltered: "+str(packed_tiles.shape))

  for j in range(packed_tiles.shape[0]):

    skip_tile = False
    if np.isnan(np.sum(packed_tiles[j])):
      skip_tile = True
    if np.isnan(np.sum(values[j])):
      skip_tile = True

    if not skip_tile:
      if len(packed_tiles_filtered)==0:
        packed_tiles_filtered = np.array([packed_tiles[j]])
        values_filtered       = np.array([values[j]])
      else:
        packed_tiles_filtered = np.append(packed_tiles_filtered,[packed_tiles[j]],axis=0)
        values_filtered       = np.append(values_filtered,[values[j]],axis=0)

  print("NaN-filtered: "+str(packed_tiles_filtered.shape))

  if i==0:
    training_tiles  = packed_tiles_filtered
    training_values = values_filtered
  else:
    training_tiles  = np.concatenate((training_tiles,packed_tiles_filtered),axis=0)
    training_values = np.concatenate((training_values,values_filtered),axis=0)

print("Training set shape: "+str(training_tiles.shape))

# RUN
# epoch is all available images
# batch is a number of non-zero tiles

g_loss = np.zeros(training_tiles.shape[0])

#for epoch in range(lastepoch,lastepoch+len(tlist)):
for epoch in range(lastepoch,500):

  print(bcolors.HEADER+"Epoch #"+str(epoch)+bcolors.ENDC)

  if os.path.isdir("result/%04d"%epoch):
    continue

  #if epoch > 2000:
  #  LR = 1e-5

  # so, here get the image, remove nans and run for 100x times
  #packed_tiles[np.isnan(packed_tiles)] = 0.0
  #tmp_vals[np.isnan(tmp_vals)] = 0.0

  input_patch = training_tiles[::2]
  gt_patch    = training_values[::2]

  st=time.time()

  #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  #run_metadata = tf.RunMetadata()
  #_,G_current,output = sess.run([G_opt,G_loss,out],feed_dict={in_tile:input_patch,gt:gt_patch,lr:LR},options=run_options,run_metadata=run_metadata)

  _,G_current,output = sess.run([G_opt,G_loss,out],feed_dict={in_tile:input_patch,gt:gt_patch,lr:LR})

  g_loss[i]=G_current
  mean_loss = np.mean(g_loss[np.where(g_loss)])

  if DEBUG_PLT_LOSS:

    recorded_loss.append(G_current)
    recorded_mean_loss.append(mean_loss)

    recorded_pr_d.append(output[0,0])
    recorded_pr_c.append(output[0,1])

    recorded_gt_d.append(gt_patch[0,0])
    recorded_gt_c.append(gt_patch[0,1])

    plt.clf()

    plt.subplot(311)

    plt.plot(recorded_loss,  label='loss')
    plt.plot(recorded_mean_loss,  label='mean loss', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title("Loss=%.5f, Mean Loss=%.5f"%(G_current,mean_loss), fontdict={'size': 20, 'color': 'red'})
    #plt.text(0.5, 0.5, 'Loss=%.5f' % G_current, fontdict={'size': 20, 'color': 'red'})

    plt.subplot(312)

    plt.xlabel('Iteration')
    plt.ylabel('Disparities')
    plt.plot(recorded_gt_d,  label='gt_d',color='green')
    plt.plot(recorded_pr_d,  label='pr_d',color='red')
    plt.legend(loc='best',ncol=1)

    plt.subplot(313)

    plt.xlabel('Iteration')
    plt.ylabel('Confidences')
    plt.plot(recorded_gt_c,  label='gt_c',color='green')
    plt.plot(recorded_pr_c,  label='pr_c',color='red')
    plt.legend(loc='best',ncol=1)

    plt.pause(0.001)

  else:
    print("%d %d Loss=%.3f CurrentLoss=%.3f Time=%.3f"%(epoch,i,mean_loss,G_current,time.time()-st))


  if epoch%save_freq==0:
    if not os.path.isdir(result_dir + '%04d'%epoch):
      os.makedirs(result_dir + '%04d'%epoch)

  saver.save(sess, checkpoint_dir + 'model.ckpt')

print_time()
print(bcolors.OKGREEN+"time: "+str(time.time())+bcolors.ENDC)

plt.ioff()
plt.show()
