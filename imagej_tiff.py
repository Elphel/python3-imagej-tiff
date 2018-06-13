#!/usr/bin/env python3

'''
/**
 * @file imagej_tiff.py
 * @brief open multi layer tiff files, display layers and parse meta data
 * @par <b>License</b>:
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
'''

__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "oleg@elphel.com"

'''
  Notes:
    - Pillow 5.1.0. Version 4.1.1 throws error (VelueError):
      ~$ (sudo) pip3 install Pillow --upgrade
      ~$ python3
      >>> import PIL
      >>> PIL.PILLOW_VERSION
      '5.1.0'
'''

from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

import sys
import xml.dom.minidom as minidom

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

# reshape to tiles
def get_tile_images(image, width=8, height=8):
  _nrows, _ncols, depth = image.shape
  _size = image.size
  _strides = image.strides

  nrows, _m = divmod(_nrows, height)
  ncols, _n = divmod(_ncols, width)
  if _m != 0 or _n != 0:
    return None

  return np.lib.stride_tricks.as_strided(
    np.ravel(image),
    shape=(nrows, ncols, height, width, depth),
    strides=(height * _strides[0], width * _strides[1], *_strides),
    writeable=False
  )

# TiffFile has no len exception
#import imageio

#from libtiff import TIFF
'''
Description:
    Reads a tiff files with multiple layers that were saved by imagej
Methods:
    .getstack(items=[])
        returns np.array, layers are stacked along depth - think of RGB channels
        @items - if empty = all, if not - items[i] - can be layer index or layer's label name
    .channel(index)
        returns np.array of a single layer
    .show_images(items=[])
        @items - if empty = all, if not - items[i] - can be layer index or layer's label name
    .show_image(index)
Examples:
#1

'''
class imagej_tiff:

  # imagej stores labels lengths in this tag
  __TIFF_TAG_LABELS_LENGTHS = 50838
  # imagej stores labels conents in this tag
  __TIFF_TAG_LABELS_STRINGS = 50839

  # init
  def __init__(self,filename):
    # file name
    self.fname = filename
    tif = Image.open(filename)
    # total number of layers in tiff
    self.nimages = tif.n_frames
    # labels array
    self.labels = []
    # infos will contain xml data Elphel stores in some of tiff files
    self.infos = []
    # dictionary from decoded infos[0] xml data
    self.props = {}

    # bits per sample, type int
    self.bpp = tif.tag[258][0]

    self.__split_labels(tif.n_frames,tif.tag)
    self.__parse_info()
    # image layers stacked along depth - (think RGB)
    self.image = []

    # fill self.image
    for i in range(self.nimages):
      tif.seek(i)
      a = np.array(tif)
      a = np.reshape(a,(a.shape[0],a.shape[1],1))

      #a = a[:,:,np.newaxis]

      # scale for 8-bits
      # exclude layer named 'other'
      if self.bpp==8:
        _min = self.data_min
        _max = self.data_max
        _MIN = 1
        _MAX = 255
        a = a.astype(float)
        if self.labels[i]!='other':
          a[a==0]=np.nan
          a = (_max-_min)*(a-_MIN)/(_MAX-_MIN)+_min

      # init
      if i==0:
        self.image = a
      # stack along depth (think of RGB channels)
      else:
        self.image = np.append(self.image,a,axis=2)

    # init done, close the image
    tif.close()

  # label == tiff layer name
  def getvalues(self,label=""):
    l = self.getstack([label],shape_as_tiles=True)
    res = np.empty((l.shape[0],l.shape[1],3))

    for i in range(res.shape[0]):
      for j in range(res.shape[1]):
        # 9x9 -> 81x1
        m = np.ravel(l[i,j])
        if self.bpp==32:
          res[i,j,0] = m[0]
          res[i,j,1] = m[2]
          res[i,j,2] = m[4]
        elif self.bpp==8:
          res[i,j,0] = ((m[0]-128)*256+m[1])/128
          res[i,j,1] = ((m[2]-128)*256+m[3])/128
          res[i,j,2] = (m[4]*256+m[5])/65536.0
        else:
          res[i,j,0] = np.nan
          res[i,j,1] = np.nan
          res[i,j,2] = np.nan

    return res


  # get ordered stack of images by provided items
  # by index or label name
  def getstack(self,items=[],shape_as_tiles=False):
    a = ()
    if len(items)==0:
      b = self.image
    else:
      for i in items:
        if type(i)==int:
          a += (self.image[:,:,i],)
        elif type(i)==str:
          j = self.labels.index(i)
          a += (self.image[:,:,j],)
      # stack along depth
      b = np.stack(a,axis=2)

    if shape_as_tiles:
      b = get_tile_images(b,self.tileW,self.tileH)

    return b

  # get np.array of a channel
  # * do not handle out of bounds
  def channel(self,index):
      return self.image[:,:,index]


  # display images by index or label
  def show_images(self,items=[]):

    # show listed only
    if len(items)>0:
      for i in items:
        if type(i)==int:
          self.show_image(i)
        elif type(i)==str:
          j = self.labels.index(i)
          self.show_image(j)
    # show all
    else:
      for i in range(self.nimages):
        self.show_image(i)


  # display single image
  def show_image(self,index):

    # display using matplotlib

    t = self.image[:,:,index]
    mytitle = "("+str(index+1)+" of "+str(self.nimages)+") "+self.labels[index]
    fig = plt.figure()
    fig.canvas.set_window_title(self.fname+": "+mytitle)
    fig.suptitle(mytitle)
    #plt.imshow(t,cmap=plt.get_cmap('gray'))
    plt.imshow(t)
    plt.colorbar()

    # display using Pillow - need to scale

    # remove NaNs - no need
    #t[np.isnan(t)]=np.nanmin(t)
    # scale to [min/max*255:255] range
    #t = (1-(t-np.nanmax(t))/(t-np.nanmin(t)))*255
    #tmp_im = Image.fromarray(t)
    #tmp_im.show()


  # puts etrees in infoss
  def __parse_info(self):

    infos = []
    for info in self.infos:
      infos.append(ET.fromstring(info))

    self.infos = infos

    # specifics
    # properties dictionary
    pd = {}

    for child in infos[0]:
      #print(child.tag+"::::::"+child.text)
      pd[child.tag] = child.text

    self.props = pd

    # tiles are squares
    self.tileW = int(self.props['tileWidth'])
    self.tileH = int(self.props['tileWidth'])
    self.data_min = float(self.props['data_min'])
    self.data_max = float(self.props['data_max'])

  # makes arrays of labels (strings) and unparsed xml infos
  def __split_labels(self,n,tag):

    # list
    tag_lens = tag[self.__TIFF_TAG_LABELS_LENGTHS]
    # string
    tag_labels = tag[self.__TIFF_TAG_LABELS_STRINGS].decode()
    # remove 1st element: it's something like IJIJlabl..
    tag_labels = tag_labels[tag_lens[0]:]
    tag_lens = tag_lens[1:]

    # the last ones are images labels
    # normally the difference is expected to be 0 or 1
    skip = len(tag_lens) - n

    self.labels = []
    self.infos = []
    for l in tag_lens:
      string = tag_labels[0:l].replace('\x00','')
      if skip==0:
        self.labels.append(string)
      else:
        self.infos.append(string)
        skip -= 1
      tag_labels = tag_labels[l:]


#MAIN
if __name__ == "__main__":

  try:
    fname = sys.argv[1]
  except IndexError:
    fname = "1521849031_093189-ML_DATA-32B-O-OFFS1.0.tiff"
    fname = "1521849031_093189-ML_DATA-08B-O-OFFS1.0.tiff"

  #fname = "1521849031_093189-DISP_MAP-D0.0-46.tif"
  #fname = "1526905735_662795-ML_DATA-08B-AIOTD-OFFS2.0.tiff"
  #fname = "test.tiff"

  print(bcolors.BOLDWHITE+"time: "+str(time.time())+bcolors.ENDC)

  ijt = imagej_tiff(fname)

  print(bcolors.BOLDWHITE+"time: "+str(time.time())+bcolors.ENDC)

  print(ijt.labels)
  print(ijt.infos)

  rough_string = ET.tostring(ijt.infos[0], "utf-8")
  reparsed = minidom.parseString(rough_string)
  print(reparsed.toprettyxml(indent="\t"))

  print(ijt.props)

  # needed properties:
  print("Tiles shape: "+str(ijt.tileW)+"x"+str(ijt.tileH))
  print("Data min: "+str(ijt.data_min))
  print("Data max: "+str(ijt.data_max))

  print(ijt.image.shape)

  # layer order: ['diagm-pair', 'diago-pair', 'hor-pairs', 'vert-pairs', 'other']
  # now split this into tiles:

  #tiles = get_tile_images(ijt.image,ijt.tileW,ijt.tileH)
  #print(tiles.shape)

  tiles = ijt.getstack(['diagm-pair','diago-pair','hor-pairs','vert-pairs'],shape_as_tiles=True)
  print("Stack of images shape: "+str(tiles.shape))

  print(bcolors.BOLDWHITE+"time: "+str(time.time())+bcolors.ENDC)
  # provide layer name
  values = ijt.getvalues(label='other')
  print("Stack of values shape: "+str(values.shape))

  print(bcolors.BOLDWHITE+"time: "+str(time.time())+bcolors.ENDC)
  #print(values)

  #print(value_tiles[131,162].flatten())
  #print(np.ravel(value_tiles[131,162]))

  #values = np.empty((vt.shape[0],vt.shape[1],3))

  #for i in range(values.shape[0]):
  #  for j in range(values.shape[1]):
  #    values[i,j,0] = get_v1()


  #print(tiles[121,160,:,:,0].shape)
  #_nrows = int(ijt.image.shape[0] / ijt.tileH)
  #_ncols = int(ijt.image.shape[1] / ijt.tileW)
  #_nrows = 32
  #_ncols = 32
  #print(str(_nrows)+" "+str(_ncols))
  #fig, ax = plt.subplots(nrows=_nrows, ncols=_ncols)
  #for i in range(_nrows):
  #  for j in range(_ncols):
  #    ax[i,j].imshow(tiles[i+100,j,:,:,0])
  #    ax[i,j].set_axis_off()

  #for i in range(5):
  #  fig = plt.figure()
  #  plt.imshow(tiles[121,160,:,:,i])
  #  plt.colorbar()

  #ijt.show_images(['other'])

  #ijt.show_images([0,3])
  #ijt.show_images(['X-corr','Y-corr'])
  #ijt.show_images(['R-vign',3])

  #ijt.show_images()
  #plt.show()





  # Examples

  # 1: get default stack of images
  #a = ijt.getstack()
  #print(a.shape)

  # 2: get defined ordered stack of images by tiff image index or by label name
  #a = ijt.getstack([1,2,'X-corr'])
  #print(a.shape)

  # 3: will throw an error if there's no such label
  #a = ijt.getstack([1,2,'Unknown'])
  #print(a.shape)

  # 4: will throw an error if index is out of bounds
  #a = ijt.getstack([1,2,'X-corr'])
  #print(a.shape)

  # 5: dev excercise
  #a = np.array([[1,2],[3,4]])
  #b = np.array([[5,6],[7,8]])
  #c = np.array([[10,11],[12,13]])

  #print("test1:")
  #ka = (a,b,c)
  #d = np.stack(ka,axis=2)

  #print(d)

  #print("test2:")
  #e = np.stack((d[:,:,1],d[:,:,0]),axis=2)
  #print(e)






