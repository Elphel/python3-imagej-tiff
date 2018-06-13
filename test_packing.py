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

  # hard coded layers names
  #lst = ['diagm-pair','diago-pair']
  #lut['diagm-pair']
  #stack = ijt.getstack(lst)
  #out = []
  #for item in lst:
  #  layer = pack_layer(tile[labels.index(item)],lut[item])
  #  out.append(layer)
  #return out


def get_packed_square(tiles,values,lut,i,j,radius):

  out = np.array([])
  # max
  Y = tiles.shape[0]
  X = tiles.shape[1]

  #print("Number of elements: "+str(2*radius+1))

  #for k in range(1):
  # print(k)

  for k in range(2*radius+1):

    y = i+k-radius
    if   y<0:
      y = 0
    elif y>(Y-1):
      y = Y-1

    for l in range(2*radius+1):

      x = j+l-radius
      if   x<0:
        x = 0
      elif x>(X-1):
        x = X-1

      #print(str(k)+" "+str(l))

      # tiles[y,x] and lut are with layers
      packed_tile = pack_tile(tiles[y,x],lut)
      #packed_tile = np.array([])
      out = np.append(out,packed_tile)

  return out


# VARS

# tiff name
tiff_name = "1521849031_093189-ML_DATA-08B-O-OFFS1.0.tiff"
# packing table name
ptab_name = "tile_packing_table.xml"

# CONSTANTS

RADIUS = 1
LAYERS_OF_INTEREST = ['diagm-pair','diago-pair']

# MAIN

# get packing table
pt = pile.PackingTable(ptab_name,LAYERS_OF_INTEREST).lut

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

l_packed = pile.pack_layer(l,pt[0])

#print(l_packed.shape)
#print(l_packed)

# a few assertions
assert l_packed[0]==(l[0,2]*1.0+l[0,3]*1.0+l[0,4]*1.0+l[0,5]*1.0+l[0,6]*1.0)
assert l_packed[1]==(l[1,1]*1.0+l[1,2]*1.0+l[2,1]*1.0+l[2,2]*1.0)
assert l_packed[15]==(l[4,4]*1.0)

print("Test: pack_layer() ... ok")

l1 = l
l2 = l*2


ls = np.dstack((l1,l2))
#equivalent: ls = np.stack((l1,l2),axis=2)
#print(ls)
#print(ls.shape)

l_packed = pile.pack_tile(ls,pt)

#print(l_packed)

# a few assertions for layer 2
assert l_packed[25]==(ls[0,2,1]*1.0+ls[0,3,1]*1.0+ls[0,4,1]*1.0+ls[0,5,1]*1.0+ls[0,6,1]*1.0)
assert l_packed[26]==(ls[1,1,1]*1.0+ls[1,2,1]*1.0+ls[2,1,1]*1.0+ls[2,2,1]*1.0)
assert l_packed[40]==(ls[4,4,1]*1.0)

print("Test: pack_tile() ... ok")

#for i in range(tiles.shape[0]):
#  for j in range(tiles.shape[1]):
    #print("tile: "+str(i)+", "+str(j))
#    FEED = get_packed_square(tiles,values,pt,i,j,RADIUS)

#xy = [(x,y) for x in range(tiles.shape[0]) for y in range(tiles.shape[1])]
#for x,y in xy:
#  FEED = get_packed_square(tiles,values,pt,x,y,RADIUS)

#def gps(i,j):

  #a = i

  #for k in range(3):
    #for l in range(3):
      #a += k+l

  #return a

#for x,y in itertools.product(range(3*tiles.shape[0]),range(3*tiles.shape[1])):
#  print(str(x)+" "+str(y))
  #FEED = get_packed_square(tiles,values,pt,x,y,RADIUS)
  #FEED = gps(x,y)

#for i in range(tiles.shape[0]):
#  for j in range(tiles.shape[1]):
#    b = gps(i,j)


#for i in range(tiles.shape[0]*tiles.shape[1]):
#  b = gps(i,i)
# basic test
#print(tiles[150,100])
#print(values[150,100])
#FEED = get_packed_square(tiles,values,pt,150,100,RADIUS)
#print(FEED)
#print(FEED.shape)

print("Done")



































































