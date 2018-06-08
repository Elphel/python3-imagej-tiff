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

import json
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import ast

def calc_packed_value(layer,index,tile):

  t = tile.flatten()
  a = layer[index]

  res = 0
  for i in a:
    res += t[i[0]]*i[1]

  return res


def iterate_tile(lut,tile):
  # hard coded layers names
  lst = ['diagm-pair','diago-pair']
  #lut['diagm-pair']
  #stack = ijt.getstack(lst)




e = ET.parse('tile_packing_table.xml').getroot()
#print(ET.tostring(e))
#reparsed = minidom.parseString(ET.tostring(e,""))
#print(reparsed.toprettyxml(indent="\t"))

# LUT is a dict
LUT = {}

for table in e:

  layer = table.get('layer')
  LUT[layer] = []

  for row in table:
    LUT[layer].append(ast.literal_eval(row.text))

print(LUT)
print("Done")
