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

from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

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
      # init
      if i==0:
        self.image = a
      # stack along depth (think of RGB channels)
      else:
        self.image = np.append(self.image,a,axis=2)

    # init done, close the image
    tif.close()

  # get ordered stack of images by provided items
  # by index or label name
  def getstack(self,items=[]):
    a = ()
    if len(items)==0:
      return self.image
    else:
      for i in items:
        if type(i)==int:
          a += (self.image[:,:,i],)
        elif type(i)==str:
          j = self.labels.index(i)
          a += (self.image[:,:,j],)
      # stack along depth
      return np.stack(a,axis=2)


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


  # puts etrees in infos
  def __parse_info(self):

    infos = []
    for info in self.infos:
      infos.append(ET.fromstring(info))

    self.infos = infos


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
  #fname = "1521849031_093189-DISP_MAP-D0.0-46.tif"
  fname = "test.tiff"

  ijt = imagej_tiff(fname)

  print(ijt.labels)
  print(ijt.infos)
  print(ijt.image.shape)

  #ijt.show_images()

  #ijt.show_images([0,3])
  ijt.show_images(['X-corr','Y-corr'])
  #ijt.show_images(['R-vign',3])

  plt.show()

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






