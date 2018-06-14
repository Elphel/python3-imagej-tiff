#!/usr/bin/env python3

__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "oleg@elphel.com"

'''
Open all tiffs in a folder, combine a single tiff from randomly selected
tiles from originals
'''
import tensorflow as tf
#import tensorflow.contrib.slim as slim

from PIL import Image

import os
import sys
import glob

import imagej_tiff as ijt
import pack_tile as pile

import numpy as np
import itertools

import time

sys.exit()

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

try:
  src = sys.argv[1]
except IndexError:
  src = "."

print_time()

tlist = glob.glob(src+"/*.tiff")

print("Found "+str(len(tlist))+" preprocessed tiff files:")
print("\n".join(tlist))
print_time()
''' WARNING, assuming:
      - timestamps and part of names match
      - layer order and names are identical
'''

# open the first one to get dimensions and other info
tiff = ijt.imagej_tiff(tlist[0])
#del tlist[0]

# shape as tiles? make a copy or make writeable
# (242, 324, 9, 9, 5)

# get labels
labels = tiff.labels.copy()
labels.remove(VALUES_LAYER_NAME)

print("Image data layers:  "+str(labels))
print("Layers of interest: "+str(LAYERS_OF_INTEREST))
print("Values layer: "+str([VALUES_LAYER_NAME]))

# create copies
tiles  = np.copy(tiff.getstack(labels,shape_as_tiles=True))
values = np.copy(tiff.getvalues(label=VALUES_LAYER_NAME))

#gt = values[:,:,1:3]

print("Mixed tiled input data shape: "+str(tiles.shape))
#print_time()

# now generate a layer of indices to get other tiles
indices = np.random.random_integers(0,len(tlist)-1,size=(tiles.shape[0],tiles.shape[1]))

#print(indices.shape)

# counts tiles from a certain tiff
shuffle_counter = np.zeros(len(tlist),np.int32)
shuffle_counter[0] = tiles.shape[0]*tiles.shape[1]

for i in range(1,len(tlist)):
  #print(tlist[i])
  tmp_tiff  = ijt.imagej_tiff(tlist[i])
  tmp_tiles = tmp_tiff.getstack(labels,shape_as_tiles=True)
  tmp_vals  = tmp_tiff.getvalues(label=VALUES_LAYER_NAME)
  #tmp_tiles =
  #tiles[indices==i] = tmp_tiff[indices==i]

  # straight and clear
  # can do quicker?
  for y,x in itertools.product(range(indices.shape[0]),range(indices.shape[1])):
    if indices[y,x]==i:
      tiles[y,x]  = tmp_tiles[y,x]
      values[y,x] = tmp_vals[y,x]
      shuffle_counter[i] +=1

# check shuffle counter
for i in range(1,len(shuffle_counter)):
  shuffle_counter[0] -= shuffle_counter[i]

print("Tiff files parts count in the mixed input = "+str(shuffle_counter))
print_time()

# test later

# now pack from 9x9 to 1x25
# tiles and values

# Parse packing table
# packing table name
ptab_name = "tile_packing_table.xml"
ptab = pile.PackingTable(ptab_name,LAYERS_OF_INTEREST).lut

# might not need it because going to loop through anyway
packed_tiles = np.array([[pile.pack_tile(tiles[i,j],ptab) for j in range(tiles.shape[1])] for i in range(tiles.shape[0])])

packed_tiles = np.dstack((packed_tiles,values[:,:,0]))

print("Packed (81x4 -> 1x(25*4+1)) tiled input shape: "+str(packed_tiles.shape))
print_time()

#for i in range(tiles.shape[0]):
#  for j in range(tiles.shape[1]):
#    nn_input = pile.get_tile_with_neighbors(tiles,i,j,RADIUS)
#    print("tile: "+str(i)+", "+str(j)+": shape = "+str(nn_input.shape))
#print_time()

result_dir = './result/'
save_freq = 500

def lrelu(x):
    return tf.maximum(x*0.2,x)

def network(input):

  fc1 = slim.fully_connected(input,42,activation_fn=lrelu,scope='g_fc1')
  fc2 = slim.fully_connected(fc1,  21,activation_fn=lrelu,scope='g_fc2')
  fc3 = slim.fully_connected(fc2,   1,activation_fn=lrelu,scope='g_fc3')

  return fc3


sess = tf.session()

in_tile = tf.placeholder(tf.float32,[None,None,101])
gt      = tf.placeholder(tf.float32,[None,None,2])
out = network(in_tile)

G_loss = tf.reduce_mean(tf.abs(out[:,0]-gt[:,0]))
t_vars=tf.trainable_variables()
lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss,var_list=[var for var in t_vars if var.name.startswith('g_')])

saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)

if ckpt:
  print('loaded '+ckpt.model_checkpoint_path)
  saver.restore(sess,ckpt.model_checkpoint_path)


allfolders = glob.glob('./result/*0')
lastepoch = 0
for folder in allfolders:
  lastepoch = np.maximum(lastepoch, int(folder[-4:]))

#g_loss = np.zeros((,1))

learning_rate = 1e-4
for epoch in range(lastepoch,4001):
  if os.path.isdir("result/%04d"%epoch):
    continue
  cnt=0
  if epoch > 2000:
    learning_rate = 1e-5

  for ind in np.random.permutation(tiles.shape[0]*tiles.shape[1]):

    input_patch = tiles[i,j]
    gt_patch = values[i,j,1:2]





