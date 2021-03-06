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
TILE_PACKING_TYPE = 1

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

result_dir = './result/'
checkpoint_dir = './result/'
save_freq = 500

def lrelu(x):
    return tf.maximum(x*0.2,x)

def network(input):

  fc1  = slim.fully_connected(input,1024,activation_fn=lrelu,scope='g_fc1')
  fc2  = slim.fully_connected(fc1,    2,activation_fn=lrelu,scope='g_fc2')
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

gt      = tf.placeholder(tf.float32,[None,2])
out = network(in_tile)

#G_loss = tf.reduce_mean(tf.abs(out[:,0]-gt[:,0]))
#t_vars=tf.trainable_variables()
#lr=tf.placeholder(tf.float32)
#G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss,var_list=[var for var in t_vars if #var.name.startswith('g_')])

saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
  print('loaded '+ckpt.model_checkpoint_path)
  saver.restore(sess,ckpt.model_checkpoint_path)

# do not need output for now
#if not os.path.isdir(result_dir + 'final/'):
#    os.makedirs(result_dir + 'final/')

tlist = glob.glob(src+"/*.tiff")

print("\n".join(tlist))
print("Found "+str(len(tlist))+" preprocessed tiff files:")
print_time()
''' WARNING, assuming:
      - timestamps and part of names match
      - layer order and names are identical
'''

# Now PROCESS
for item in tlist:
  print(bcolors.OKGREEN+"Processing "+item+bcolors.ENDC)
  # open the first one to get dimensions and other info
  tiff = ijt.imagej_tiff(item)

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
  # need values? Well, just to compare
  values = np.copy(tiff.getvalues(label=VALUES_LAYER_NAME))
  #gt = values[:,:,1:3]

  print("Mixed tiled input data shape: "+str(tiles.shape))
  #print_time()

  # now pack from 9x9 to 1x25
  # tiles and values

  # might not need it because going to loop through anyway
  if   TILE_PACKING_TYPE==1:
    packed_tiles = pile.pack(tiles)
  elif TILE_PACKING_TYPE==2:
    packed_tiles = pile.pack(tiles,TILE_PACKING_TYPE)

  packed_tiles = np.dstack((packed_tiles,values[:,:,0]))

  print(packed_tiles.shape)
  print("ENDDD!")

  # NO
  # flatten
  #packed_tiles_flat = packed_tiles.reshape(-1, packed_tiles.shape[-1])
  #values_flat       = values.reshape(-1, values.shape[-1])

  #print("Packed (81x4 -> 1x(25*4+1)) tiled input shape: "+str(packed_tiles_flat.shape))
  #print("Values shape "+str(values_flat.shape))
  #print_time()

  # do line by line?!
  output_image = np.empty((packed_tiles.shape[0],packed_tiles.shape[1],2))
  print("Output shape = "+str(output_image.shape))

  for i in range(packed_tiles.shape[0]):

    # now run prediction
    packed_tiles_flat = packed_tiles[i]
    values_flat       = values[i]

    # whole row at once
    output = sess.run(out,feed_dict={in_tile:packed_tiles_flat})
    output_image[i] = output

    # so, let's print
    for j in range(packed_tiles.shape[0]):
      p  = output[j,0]
      pc = output[j,1]
      fv = values_flat[j,0]
      gt = values_flat[j,1]
      cf = values_flat[j,2]

      vstring = "["+"{0:.2f}".format(fv)+", "+"{0:.2f}".format(gt)+", "+"{0:.2f}".format(cf)+"]"
      pstring = "["+"{0:.2f}".format(p)+", "+"{0:.2f}".format(pc)+"]"

      if not np.isnan(p):
        outstring = "i,j: "+str(i)+"  "+str(j)+"    Values:  "+vstring+"    Prediction:  "+pstring
        if abs(cf)<0.5:
          print(outstring)
          #pass
        else:
          print(bcolors.WARNING+outstring+bcolors.ENDC)


  sess.close()
  import imagej_tiffwriter

  # 1 prediction
  # 2 ground truth
  # difference 1 - 2

  im1 = output_image[:,:,0]
  im2 = values[:,:,1]
  im3 = im1-im2

  tif = np.dstack((im1,im2,im3))

  im3 = np.ravel(im3)

  print(im3.shape)

  im4 = im3[~np.isnan(im3)]

  rms = np.sqrt(np.mean(np.square(im4)))
  print("RMS = "+str(rms))

  imagej_tiffwriter.save('prediction_results.tiff',tif)

  #sys.exit(0)

    #else:
    #  print("i: "+str(i)+" NaNs")

# check histogram:


#import matplotlib.pyplot as plt

#x = np.histogram(values_flat[:,2])
#plt.hist(x, bins=100)
#plt.ylabel('Confidence')

print_time()
print(bcolors.OKGREEN+"time: "+str(time.time())+bcolors.ENDC)































