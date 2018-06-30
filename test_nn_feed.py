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

print("Importing TensorCrawl")
print_time()

import tensorflow as tf
import tensorflow.contrib.slim as slim

print("TensorCrawl imported")
print_time()

IS_TEST = False

# BEGIN IF IS_TEST
if not IS_TEST:

  tlist = glob.glob(src+"/*.tiff")

  print("\n".join(tlist))
  print("Found "+str(len(tlist))+" preprocessed tiff files:")
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
  print("Values shape "+str(values.shape))
  print_time()

else:
  print("Init test data")

  ptab_name = "tile_packing_table.xml"
  pt = pile.PackingTable(ptab_name,LAYERS_OF_INTEREST).lut

  # 9x9 2 layers, no neighbors
  l = np.zeros((9,9))
  for y,x in itertools.product(range(l.shape[0]),range(l.shape[1])):
    l[y,x] = 9*y + x

  l_value = np.array([2.54,3.54,0.5])

  #print(l)

  l1 = l
  l2 = l*2
  l3 = l*3
  l4 = l*4

  ls = np.dstack((l1,l2,l3,l4))
  #print(ls.shape)

  l_packed_pre = pile.pack_tile(ls,pt)
  #print(l_packed_pre.shape)
  #print(l_packed_pre)

  l_packed = np.hstack((l_packed_pre,l_value[0]))

  #print(l_packed.shape)
  #print(l_packed)
  # use l_packed

  packed_tiles = np.empty([1,1,l_packed.shape[0]])
  values =       np.empty([1,1,2])
  print(packed_tiles.shape)
  print(values.shape)

  packed_tiles[0,0] = l_packed
  values[0,0] = l_value[1:3]

  print(packed_tiles[0,0])
  print(values[0,0])


# END IF IS_TEST


#print("CHECKPOINTE")

#for i in range(tiles.shape[0]):
#  for j in range(tiles.shape[1]):
#    nn_input = pile.get_tile_with_neighbors(tiles,i,j,RADIUS)
#    print("tile: "+str(i)+", "+str(j)+": shape = "+str(nn_input.shape))
#print_time()

result_dir = './result/'
checkpoint_dir = './result/'
save_freq = 500

def lrelu(x):
    return tf.maximum(x*0.2,x)

def network(input):

  fc1 = slim.fully_connected(input,101,activation_fn=lrelu,scope='g_fc1')
  fc2 = slim.fully_connected(fc1,  101,activation_fn=lrelu,scope='g_fc2')
  fc3 = slim.fully_connected(fc2,  101,activation_fn=lrelu,scope='g_fc3')
  fc4 = slim.fully_connected(fc3,  101,activation_fn=lrelu,scope='g_fc4')
  fc5 = slim.fully_connected(fc4,    2,activation_fn=lrelu,scope='g_fc5')

  return fc5


sess = tf.Session()

in_tile = tf.placeholder(tf.float32,[None,101])
gt      = tf.placeholder(tf.float32,[None,2])
#cf_cutoff = tf.constant(tf.float32,[None,1])
out = network(in_tile)

# min cutoff
cf_cutoff = 0.173303


cf_w = tf.pow(tf.maximum(gt[:,1]-cf_cutoff,0.0),1)
cf_wsum = tf.reduce_sum(cf_w)
cf_w_norm = cf_w/cf_wsum

#out_cf = out[:,1]

G_loss = tf.reduce_mean(tf.abs(out[:,0]-cf_w_norm*gt[:,0]))

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

g_loss = np.zeros((packed_tiles.shape[0]*packed_tiles.shape[1],1))

learning_rate = 1e-4

print(bcolors.HEADER+"Last Epoch = "+str(lastepoch)+bcolors.ENDC)

for epoch in range(lastepoch,1):
#for epoch in range(lastepoch,4001):
  if os.path.isdir("result/%04d"%epoch):
    continue
  cnt=0
  if epoch > 2000:
    learning_rate = 1e-5

  for ind in np.random.permutation(packed_tiles.shape[0]*packed_tiles.shape[1]):

    #print("Iteration "+str(cnt))

    st=time.time()
    cnt+=1

    i = int(ind/packed_tiles.shape[1])
    j = ind%packed_tiles.shape[1]
    #input_patch = tiles[i,j]
    input_patch = np.empty((1,packed_tiles.shape[2]))
    input_patch[0] = packed_tiles[i,j]

    gt_patch = np.empty((1,2))

    if not IS_TEST:
      gt_patch[0] = values[i,j,1:3]
    else:
      gt_patch[0] = values[i,j]

    #print(input_patch)
    #print(gt_patch)

    gt_patch[gt_patch==-256] = np.nan

    skip_iteration = False
    # if nan skip run!
    if np.isnan(np.sum(gt_patch[0])):
      skip_iteration = True

    if np.isnan(np.sum(input_patch[0])):
      skip_iteration = True


    if skip_iteration:
      #print(bcolors.WARNING+"Found NaN, skipping iteration for tile "+str(i)+","+str(j)+bcolors.ENDC)
      pass
    else:
      _,G_current,output = sess.run([G_opt,G_loss,out],feed_dict={in_tile:input_patch,gt:gt_patch,lr:learning_rate})
      g_loss[ind]=G_current

      print("%d %d Loss=%.3f CurrentLoss=%.3f Time=%.3f"%(epoch,cnt,np.mean(g_loss[np.where(g_loss)]),G_current,time.time()-st))

    if epoch%save_freq==0:
      if not os.path.isdir(result_dir + '%04d'%epoch):
        os.makedirs(result_dir + '%04d'%epoch)

  saver.save(sess, checkpoint_dir + 'model.ckpt')

print_time()
print(bcolors.OKGREEN+"time: "+str(time.time())+bcolors.ENDC)


