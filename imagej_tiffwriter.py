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

from PIL import Image, TiffImagePlugin
import numpy as np
import math

def __get_IJ_IFD(t,z,c):

  ifd = TiffImagePlugin.ImageFileDirectory_v2()
  #is_hyperstack = 'true' if len(shape)>1 else 'false'
  #if (len(shape)>0):
  return ifd


def save(path,images,force_stack=False,force_hyperstack=False):

  # Got images, analyze shape:
  #   - possible formats (c == depth):
  #     -- (t,z,h,w,c)
  #     -- (t,h,w,c), t or z does not matter
  #     -- (h,w,c)
  #     -- (h,w)

  # save single layer image
  if   len(images.shape)==2:
    # (h,w) -> (h,w,c=1)
    images = images[:,:,np.newaxis]
  elif len(images.shape)>2:
    # do nothing
    pass
  else:
    # (w,) -> (h=1,w,c=1)
    images = images[np.newaxis,:,np.newaxis]

  #imlist = np.ravel(Image.fromarray(images))
  t,z,h,w,c  = images.shape
  c_axis = len(images.shape)-1
  channels = np.squeeze(np.split(images,c,axis=c_axis))

  split_channels = np.concatenate(channels,axis=-3)
  images_flat = np.reshape(split_channels,(-1,h,w))

  imlist = []
  for i in range(images_flat.shape[0]):
    imlist.append(Image.fromarray(images_flat[i]))

  imlist[0].save(path,save_all=True,append_images=imlist[1:],tiffinfo=__get_IJ_IFD(t,z,c))

# Testing
if __name__ == "__main__":


  def hamming_window(x,N):
    y = 0.54 - 0.46*math.cos(2*math.pi*x/(N-1))
    return y

  hw = hamming_window

  NT = 5
  NC = 2
  NZ = 3
  NX = 512
  NY = 512

  images = np.empty((NT,NZ,NY,NX,NC))

  import time
  print(str(time.time())+": Generating test images")
  for t in range(NT):
    for z in range(NZ):
      for c in range(NC):
        images[t,z,:,:,c] = np.array([[(255-t*25)*hw(i,512)*hw(j,512) for i in range(NX)] for j in range(NY)],np.float32)
  print(str(time.time())+": Test images generated")
  print("Images shape: "+str(images.shape))

  v = save("result_2.tiff",images)
