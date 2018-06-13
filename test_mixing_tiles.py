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

import numpy as np
import itertools

# USAGE: python3 test_3.py some-path

VALUES_LAYER_NAME = 'other'

try:
  src = sys.argv[1]
except IndexError:
  src = "."

tlist = glob.glob(src+"/*.tiff")

print("Found "+str(len(tlist))+" tiff files:")
print("\n".join(tlist))

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

print("Image data layers: "+str(labels))
print("Values layer: "+str([VALUES_LAYER_NAME]))

# create copies
tiles  = np.copy(tiff.getstack(labels,shape_as_tiles=True))
tiles_bkp = np.copy(tiles)
values = np.copy(tiff.getvalues(label=VALUES_LAYER_NAME))

print("Tiled tiff shape: "+str(tiles.shape))

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

print("Tiffs shuffle counter = "+str(shuffle_counter))

# test later

# now pack from 9x9 to 1x25
# tiles and values













