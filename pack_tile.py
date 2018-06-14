#!/usr/bin/env python3

__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "oleg@elphel.com"

import numpy as np
import xml.etree.ElementTree as ET
import ast

class PackingTable:

  def __init__(self,filename,layers_of_interest):
    e = ET.parse(filename).getroot()
    #print(ET.tostring(e))
    #reparsed = minidom.parseString(ET.tostring(e,""))
    #print(reparsed.toprettyxml(indent="\t"))

    # Parse xml:
    # td = tmp_dict
    td = {}
    for table in e:
      layer = table.get('layer')
      td[layer] = []
      for row in table:
        # safe evaluation
        td[layer].append(ast.literal_eval(row.text))

    # order
    LUT = []
    for layer in layers_of_interest:
      LUT.append(td[layer])

    self.lut = LUT


# A tile consists of layers
# layer is packed from 9x9 to 25x1
def pack_layer(layer,lut_row):

  #print(layer.shape)

  t = layer.flatten()

  out = np.array([])

  # iterate through rows
  for i in range(len(lut_row)):
    val = 0
    # process row value
    for j in lut_row[i]:
      if np.isnan(t[j[0]]):
        val = np.nan
        break
      val += t[j[0]]*j[1]
    out = np.append(out,val)

  return out


# tile and lut already ordered and indices match
def pack_tile(tile,lut):

  out = np.array([])
  for i in range(len(lut)):
    layer = pack_layer(tile[:,:,i],lut[i])
    out = np.append(out,layer)

  return out

# tiles are already packed
def get_tile_with_neighbors(tiles,i,j,radius):

  out = np.array([])
  # max
  Y,X = tiles.shape[0:2]
  #print(str(Y)+" "+str(X))

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

      out = np.append(out,tiles[y,x])

  return out










