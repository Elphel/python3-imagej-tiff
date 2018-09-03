#!/usr/bin/env python3

'''
/**
 * @file imagej_tiff_saver.py
 * @brief save tiffs for imagej (1.52d+) - with stacks and hyperstacks
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
Usage example:
  import imagej_tiffwriter
  import numpy as np

  # have a few images in the form of numpy arrays
  # make sure to stack them as:
  #   - (t,z,h,w,c)
  #   - (z,h,w,c)
  #   - (h,w,c)
  #   - (h,w)

  imagej_tiffwriter.save(path,images)

'''

from PIL import Image, TiffImagePlugin
import numpy as np
import math

# DO NOT USE?
# thing is the old ImageJs <1.52d poorly handle tags directories or something like that
def __get_IJ_IFD(t,z,c):

  #ifd = TiffImagePlugin.ImageFileDirectory_v2(ifh=b"MM*\x00\x00\x00\x00\x00",prefix= b"MM")
  ifd = TiffImagePlugin.ImageFileDirectory_v2(prefix= b"MM")
  #ifd = TiffImagePlugin.ImageFileDirectory_v2()
  #ifd = TiffImagePlugin.ImageFileDirectory(prefix='MM')

  ijheader = [
    'ImageJ=',
    'hyperstack=true',
    'images='+str(t*z*c),
    'channels='+str(c),
    'slices='+str(z),
    'frames='+str(t),
    'loop=false'
  ]

  ifd[270] = ("\n".join(ijheader)+"\n")

  ijlabl = [
    b'IJIJinfo\x00\x00\x00\x01labl\x00\x00\x00\x05',
    '<info></info>'.encode('UTF-16-LE'),
    'img1'.encode('UTF-16-LE'),
    'img2'.encode('UTF-16-LE'),
    'img3'.encode('UTF-16-LE'),
    'img4'.encode('UTF-16-LE'),
    'img5'.encode('UTF-16-LE')
  ]

  #for i in range()

  #ifd[50838] = (20,10)
  #ifd[50839] = (20,10)
  ifd_50838_list = []
  ifd_50839 = b"".join(ijlabl)

  for label in ijlabl:
    ifd_50838_list.append(len(label))

  ifd_50838 = tuple(ifd_50838_list)
  #print(ifd_50838)
  #print(ifd_50839)

  #is_hyperstack = 'true' if len(shape)>1 else 'false'
  #if (len(shape)>0):

  ifd[50838] = ifd_50838
  ifd[50839] = ifd_50839

  # override
  #tif = Image.open("test.tiff")
  #tag_50838 = tif.tag[50838]
  #tag_50839 = tif.tag[50839]
  #tag_270 = tif.tag[270]

  #print(tag_50839)

  #ifd[270] = tag_270
  #ifd[50838] = tag_50838
  #ifd[50839] = tag_50839

  return ifd


#def save(path,images,force_stack=False,force_hyperstack=False):
def save(path,images):

  # Got images, analyze shape:
  #   - possible formats (c == depth):
  #     -- (t,z,h,w,c)
  #     -- (t,h,w,c), t or z does not matter
  #     -- (h,w,c)
  #     -- (h,w)


  # 0 or 1 images.shapes are not handled
  #
  # (h,w)
  if len(images.shape)==2:

    image = Image.fromarray(images)
    image.save(path)

  elif len(images.shape)>2:

    h,w,c  = images.shape[-3:]

    if len(images.shape)==3:
      images = np.reshape(images,(1,h,w,c))

    z = images.shape[-4]

    if len(images.shape)==4:
      images = np.reshape(images,(1,z,h,w,c))

    t  = images.shape[-5]

    c_axis = -1

    if c==1:
      split_channels = images
    else:
      channels = np.array(np.split(images,c,axis=c_axis))
      split_channels = np.concatenate(channels,axis=-3)

    images_flat = np.reshape(split_channels,(-1,h,w))

    imlist = []
    for i in range(images_flat.shape[0]):
      imlist.append(Image.fromarray(images_flat[i]))

    #imlist[0].save(path,save_all=True,append_images=imlist[1:])
    # thing is the old ImageJs <1.52d poorly handle tags directories or something like that
    imlist[0].save(path,save_all=True,append_images=imlist[1:],tiffinfo=__get_IJ_IFD(t,z,c))

# Testing
if __name__ == "__main__":

  def hamming_window(x,N):
    y = 0.54 - 0.46*math.cos(2*math.pi*x/(N-1))
    return y

  hw = hamming_window

  NT = 5
  NC = 1
  NZ = 1
  NX = 2916
  NY = 2178

  images = np.empty((NT,NZ,NY,NX,NC))

  import time
  print(str(time.time())+": Generating test images")
  for t in range(NT):
    for z in range(NZ):
      for c in range(NC):
        images[t,z,:,:,c] = np.array([[(255-t*25)*hw(i,512)*hw(j,512) for i in range(NX)] for j in range(NY)],np.float32)
  print(str(time.time())+": Test images generated")
  print("Images shape: "+str(images.shape))

  print("1D run")
  imgs = images[:,0,:,:,0]
  print(imgs.shape)

  imlist = []
  for i in range(imgs.shape[0]):
    tmp = Image.fromarray(imgs[i])
    #tmp.mode = "I;32BS"
    imlist.append(tmp)

  TiffImagePlugin.DEBUG = True
  TiffImagePlugin.WRITE_LIBTIFF = False

  #import sys
  #sys.byteorder = "big"
  #imlist[0].mode = "I;16B"
  #print("encoderconfig")
  #print(imlist[0].encoderconfig)

  imlist[0].save("result_1D.tiff",save_all=True,append_images=imlist[1:],tiffinfo=__get_IJ_IFD(NT,NZ,NC))
  #v = save("result_1D.tiff",imgs)

  #tif = Image.open("result_1D.tiff")
  #tag_50838 = tif.tag[50838]
  #tag_50839 = tif.tag[50839]
  #tag_270 = tif.tag[270]

  #print(tag_50839)

  #tif = Image.open("test.tiff")

  #tif.save("test_saved.tiff")

  #print("5D run")
  #v = save("result_5D.tiff",images)
  #print("4D run")
  #v = save("result_4D.tiff",images[0])
  #print("3D run")
  #v = save("result_3D.tiff",images[0,0])

  #print("3D run, 1 channel")
  #tmp_images = images[0,0,:,:,0]
  #v = save("result_3D1C.tiff",tmp_images[:,:,np.newaxis])

  # open test






