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

import numpy as np
import struct
import tifffile
import math

# from here: https://stackoverflow.com/questions/50258287/how-to-specify-colormap-when-saving-tiff-stack
def imagej_metadata_tags(metadata, byteorder):
    """Return IJMetadata and IJMetadataByteCounts tags from metadata dict.

    The tags can be passed to the TiffWriter.save function as extratags.

    """
    header = [{'>': b'IJIJ', '<': b'JIJI'}[byteorder]]
    bytecounts = [0]
    body = []

    def writestring(data, byteorder):
        return data.encode('utf-16' + {'>': 'be', '<': 'le'}[byteorder])

    def writedoubles(data, byteorder):
        return struct.pack(byteorder+('d' * len(data)), *data)

    def writebytes(data, byteorder):
        return data.tobytes()

    metadata_types = (
        ('Info', b'info', 1, writestring),
        ('Labels', b'labl', None, writestring),
        ('Ranges', b'rang', 1, writedoubles),
        ('LUTs', b'luts', None, writebytes),
        ('Plot', b'plot', 1, writebytes),
        ('ROI', b'roi ', 1, writebytes),
        ('Overlays', b'over', None, writebytes))

    for key, mtype, count, func in metadata_types:
        if key not in metadata:
            continue
        if byteorder == '<':
            mtype = mtype[::-1]
        values = metadata[key]
        if count is None:
            count = len(values)
        else:
            values = [values]
        header.append(mtype + struct.pack(byteorder+'I', count))
        for value in values:
            data = func(value, byteorder)
            body.append(data)
            bytecounts.append(len(data))

    body = b''.join(body)
    header = b''.join(header)
    data = header + body
    bytecounts[0] = len(header)
    bytecounts = struct.pack(byteorder+('I' * len(bytecounts)), *bytecounts)
    return ((50839, 'B', len(data), data, True),
            (50838, 'I', len(bytecounts)//4, bytecounts, True))


#def save(path,images,force_stack=False,force_hyperstack=False):
def save(path,images,labels=None,label_prefix="Label"):

  '''
    labels a list or None
  '''

  '''
    Expecting:
    (h,w),
    (n,h,w) - just create a simple stack
  '''

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
    images = images[np.newaxis,...]

  # now the shape length is 3
  if len(images.shape)==3:
    # tifffile treats shape[0] as channel, need to expand to get labels displayed
    #images = images[images.shape[0],np.newaxis,images.shape[1],images.shape[2]]
    images = np.reshape(images,(images.shape[0],1,images.shape[1],images.shape[2]))

    labels_list = []
    if labels is None:
      for i in range(images.shape[0]):
        labels_list.append(label_prefix+str(i))
    else:
      labels_list = labels

    print(labels_list)

    ijtags = imagej_metadata_tags({'Labels':labels_list}, '<')

    with tifffile.TiffWriter(path, bigtiff=False,imagej=True) as tif:
      for i in range(images.shape[0]):
        print(images[i].shape)
        tif.save(images[i], metadata={'version':'1.11a',' loop':False}, extratags=ijtags)

# Testing
if __name__ == "__main__":

  def hamming_window(x,N):
    y = 0.54 - 0.46*math.cos(2*math.pi*x/(N-1))
    return y

  hw = hamming_window

  NT = 5
  NX = 512
  NY = 512

  images = np.empty((NT,NY,NX))

  import time
  print(str(time.time())+": Generating test images")
  for t in range(NT):
    images[t,:,:] = np.array([[(255-t*25)*hw(i,512)*hw(j,512) for i in range(NX)] for j in range(NY)],np.float32)
  print(str(time.time())+": Test images generated")
  print("Images shape: "+str(images.shape))
  v = save("tiffwriter_test.tiff",images)
