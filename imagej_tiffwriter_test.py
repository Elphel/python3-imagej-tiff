#!/usr/bin/env python3

import struct
import numpy
import tifffile
import math

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


def hamming_window(x,N):
  y = 0.54 - 0.46*math.cos(2*math.pi*x/(N-1))
  return y

hw = hamming_window

image =  numpy.array([[(255-0*25)*hw(i,512)*hw(j,512) for i in range(512)] for j in range(512)],numpy.float32)
image = image[numpy.newaxis,...]

ijtags = imagej_metadata_tags({'Labels':["Name1","Name2","Name3","Name4","Name5"]}, '<')

print(ijtags)

with tifffile.TiffWriter("multipage_test.tiff", bigtiff=False,imagej=True) as tif:
  for i in range(5):
    tif.save(image, metadata={'version':'20180905', 'loop':False}, extratags=ijtags)
