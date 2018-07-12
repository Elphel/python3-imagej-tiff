#!/usr/bin/env python3
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import imagej_tiff as ijt

#tiff = ijt.imagej_tiff('test.tiff')
#print(tiff.nimages)
#print(tiff.labels)
#print(tiff.infos)
#tiff.show_images(['X-corr','Y-corr',0,2])
#plt.show()

import itertools

import pack_tile as pile

# VARS

# tiff name
tiff_name = "1521849031_093189-ML_DATA-08B-O-OFFS1.0.tiff"

# CONSTANTS

RADIUS = 1
LAYERS_OF_INTEREST = ['diagm-pair','diago-pair','hor-pairs','vert-pairs']

# MAIN

# get tiff
tiff   = ijt.imagej_tiff(tiff_name)
tiles  = tiff.getstack(LAYERS_OF_INTEREST,shape_as_tiles=True)
values = tiff.getvalues(label='other')

#tiff.show_images(LAYERS_OF_INTEREST)
#plt.show()
print(tiles.shape)

# now iterate through tiles, get neighbors

# 9x9 2 layers, no neighbors
l = np.zeros((9,9))
for y,x in itertools.product(range(l.shape[0]),range(l.shape[1])):
  l[y,x] = 9*y + x

print(l)

l2 = np.reshape(l,(1,1,9,9))
l_stack = np.stack((l2,l2,l2,l2),axis=-1)

print(l_stack.shape)

l_packed = pile.pack(l_stack)

print(l_packed.shape)

l_packed2 = np.reshape(l_packed,l_packed.shape[-1])

#print(l_packed)

# a few assertions
assert l_packed2[0]==(l[0,2]*1.0+l[0,3]*1.0+l[0,4]*1.0+l[0,5]*1.0+l[0,6]*1.0)
assert l_packed2[1]==(l[1,1]*1.0+l[1,2]*1.0+l[2,1]*1.0+l[2,2]*1.0)
assert l_packed2[15]==(l[4,4]*1.0)

print("Test: pack() ... ok")

print("Done")



































































