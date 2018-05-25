#!/usr/bin/env python3

from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

import sys
import xml.dom.minidom as minidom

import time

from imagej_tiff import imagej_tiff

fname08bpp = "1521849031_093189-ML_DATA-08B-O-OFFS1.0.tiff"
fname32bpp = "1521849031_093189-ML_DATA-32B-O-OFFS1.0.tiff"

im08 = imagej_tiff(fname08bpp)
im32 = imagej_tiff(fname32bpp)

print(im08.props)

THRESHOLD = 0.01
THRESHOLD2 = 0.1

values08 = im08.getvalues(label='other')
values32 = im32.getvalues(label='other')

# compare values

print("test1: Compare values layers")

values32[np.isnan(values32)] = 0
values08[np.isnan(values08)] = 0
values08[values08==-256.0] = 0

delta = abs(values32 - values08)
#print(delta)
#print(delta.shape)

error_counter = 0
for i in range(delta.shape[0]):
  for j in range(delta.shape[1]):
    for k in range(delta.shape[2]):
      if delta[i,j,k]>THRESHOLD:
        error_counter += 1
        print("Fail ("+str(error_counter)+"): "+str((i,j,k))+" "+str(delta[i,j,k]))
        print(str(values32[i,j,k])+" vs "+str(values08[i,j,k]))

if error_counter==0:
  print("Values are OK")

#v08 = im08.getstack(['other'],shape_as_tiles=True)
#v32 = im32.getstack(['other'],shape_as_tiles=True)

#print(v08.shape)

#_nrows = 1
#_ncols = 1

#fig = plt.figure()
#plt.imshow(v08[241,321,:,:,0])
#plt.colorbar()

#fig = plt.figure()
#plt.imshow(v32[241,321,:,:,0])
#plt.colorbar()

# compare layers

print("test2: Compare layers")

tiles08 = im08.getstack(['diagm-pair','diago-pair','hor-pairs','vert-pairs'],shape_as_tiles=True)
tiles32 = im32.getstack(['diagm-pair','diago-pair','hor-pairs','vert-pairs'],shape_as_tiles=True)

delta2 = abs(tiles32-tiles08)
error_counter = 0
for i in range(delta2.shape[0]):
  for j in range(delta2.shape[1]):
    for k in range(delta2.shape[2]):
      for l in range(delta2.shape[3]):
        for m in range(delta2.shape[4]):
          if delta2[i,j,k,l,m]>THRESHOLD2:
            error_counter += 1
            print("Fail ("+str(error_counter)+"): "+str((i,j,k,l,m))+" "+str(delta2[i,j,k,l,m]))
            print(str(tiles32[i,j,k,l,m])+" vs "+str(tiles08[i,j,k,l,m]))

if error_counter==0:
  print("Layers are OK")

#fig = plt.figure()
#plt.imshow(tiles08[121,187,:,:,1])
#plt.colorbar()

#fig = plt.figure()
#plt.imshow(tiles32[121,187,:,:,1])
#plt.colorbar()

plt.show()
























































