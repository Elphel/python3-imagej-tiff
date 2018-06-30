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

try:
  src = sys.argv[1]
except IndexError:
  src = "."


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
  #tiles  = np.copy(tiff.getstack(labels,shape_as_tiles=True))
  # need values? Well, just to compare
  values = np.copy(tiff.getvalues(label=VALUES_LAYER_NAME))
  #gt = values[:,:,1:3]

  print("Mixed tiled input data shape: "+str(values.shape))

  # plot ground truth
  values[values==-256] = np.nan

  t = values[:,:,1]
  mytitle = "Ground truth"
  fig = plt.figure()
  fig.canvas.set_window_title(mytitle)
  fig.suptitle(mytitle)
  plt.imshow(t)
  plt.colorbar()
  #plt.show()

  t = values[:,:,2]
  mytitle = "Confidence"
  fig = plt.figure()
  fig.canvas.set_window_title(mytitle)
  fig.suptitle(mytitle)
  plt.imshow(t)
  plt.colorbar()
  #plt.show()

  mytitle = "Confidence histogram"
  fig = plt.figure()
  fig.canvas.set_window_title(mytitle)
  fig.suptitle(mytitle)

  values_flat = np.ravel(values[:,:,2])
  #values_flat[values_flat==0] = np.nan

  # flat and filtered
  values_ff = values_flat[values_flat!=0]
  values_ff = np.round(values_ff,3)

  plt.hist(values_ff,bins=1000)
  plt.ylabel('N')

  plt.show()


print_time()
print(bcolors.OKGREEN+"time: "+str(time.time())+bcolors.ENDC)































