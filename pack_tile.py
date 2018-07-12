#!/usr/bin/env python3

__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "oleg@elphel.com"

import numpy as np

# pack from 9x9x4 to 25x1
def pack_tile_type1(tile):

  out = np.empty(100)

  # pack diagm-pair
  l = np.ravel(tile[:,:,0])
  out[ 0] = 1.0*l[ 2]+1.0*l[ 3]+1.0*l[ 4]+1.0*l[ 5]+1.0*l[6]
  out[ 1] = 1.0*l[10]+1.0*l[11]+1.0*l[19]+1.0*l[20]
  out[ 2] = 1.0*l[12]+1.0*l[13]+1.0*l[14]
  out[ 3] = 1.0*l[15]+1.0*l[16]+1.0*l[24]+1.0*l[25]
  out[ 4] = 1.0*l[18]+1.0*l[27]+1.0*l[36]+1.0*l[45]+1.0*l[54]
  out[ 5] = 1.0*l[21]+1.0*l[22]+1.0*l[23]
  out[ 6] = 1.0*l[26]+1.0*l[35]+1.0*l[44]+1.0*l[53]+1.0*l[62]
  out[ 7] = 1.0*l[28]+1.0*l[37]+1.0*l[46]
  out[ 8] = 1.0*l[29]+1.0*l[38]+1.0*l[47]
  out[ 9] = 1.0*l[30]
  out[10] = 1.0*l[31]
  out[11] = 1.0*l[32]
  out[12] = 1.0*l[33]+1.0*l[42]+1.0*l[51]
  out[13] = 1.0*l[34]+1.0*l[43]+1.0*l[52]
  out[14] = 1.0*l[39]
  out[15] = 1.0*l[40]
  out[16] = 1.0*l[41]
  out[17] = 1.0*l[48]
  out[18] = 1.0*l[49]
  out[19] = 1.0*l[50]
  out[20] = 1.0*l[55]+1.0*l[56]+1.0*l[64]+1.0*l[65]
  out[21] = 1.0*l[57]+1.0*l[58]+1.0*l[59]
  out[22] = 1.0*l[60]+1.0*l[61]+1.0*l[69]+1.0*l[70]
  out[23] = 1.0*l[66]+1.0*l[67]+1.0*l[68]
  out[24] = 1.0*l[74]+1.0*l[75]+1.0*l[76]+1.0*l[77]+1.0*l[78]

  # pack diago-pair
  l = np.ravel(tile[:,:,1])
  out[25] = 1.0*l[ 2]+1.0*l[ 3]+1.0*l[ 4]+1.0*l[ 5]+1.0*l[6]
  out[26] = 1.0*l[10]+1.0*l[11]+1.0*l[19]+1.0*l[20]
  out[27] = 1.0*l[12]+1.0*l[13]+1.0*l[14]
  out[28] = 1.0*l[15]+1.0*l[16]+1.0*l[24]+1.0*l[25]
  out[29] = 1.0*l[18]+1.0*l[27]+1.0*l[36]+1.0*l[45]+1.0*l[54]
  out[30] = 1.0*l[21]+1.0*l[22]+1.0*l[23]
  out[31] = 1.0*l[26]+1.0*l[35]+1.0*l[44]+1.0*l[53]+1.0*l[62]
  out[32] = 1.0*l[28]+1.0*l[37]+1.0*l[46]
  out[33] = 1.0*l[29]+1.0*l[38]+1.0*l[47]
  out[34] = 1.0*l[30]
  out[35] = 1.0*l[31]
  out[36] = 1.0*l[32]
  out[37] = 1.0*l[33]+1.0*l[42]+1.0*l[51]
  out[38] = 1.0*l[34]+1.0*l[43]+1.0*l[52]
  out[39] = 1.0*l[39]
  out[40] = 1.0*l[40]
  out[41] = 1.0*l[41]
  out[42] = 1.0*l[48]
  out[43] = 1.0*l[49]
  out[44] = 1.0*l[50]
  out[45] = 1.0*l[55]+1.0*l[56]+1.0*l[64]+1.0*l[65]
  out[46] = 1.0*l[57]+1.0*l[58]+1.0*l[59]
  out[47] = 1.0*l[60]+1.0*l[61]+1.0*l[69]+1.0*l[70]
  out[48] = 1.0*l[66]+1.0*l[67]+1.0*l[68]
  out[49] = 1.0*l[74]+1.0*l[75]+1.0*l[76]+1.0*l[77]+1.0*l[78]

  # pack hor-pairs
  l = np.ravel(tile[:,:,2])
  out[50] = 1.0*l[ 2]+1.0*l[ 3]+1.0*l[ 4]+1.0*l[ 5]+1.0*l[6]
  out[51] = 1.0*l[10]+1.0*l[11]+1.0*l[19]+1.0*l[20]
  out[52] = 1.0*l[12]+1.0*l[13]+1.0*l[14]
  out[53] = 1.0*l[15]+1.0*l[16]+1.0*l[24]+1.0*l[25]
  out[54] = 1.0*l[18]+1.0*l[27]+1.0*l[36]+1.0*l[45]+1.0*l[54]
  out[55] = 1.0*l[21]+1.0*l[22]+1.0*l[23]
  out[56] = 1.0*l[26]+1.0*l[35]+1.0*l[44]+1.0*l[53]+1.0*l[62]
  out[57] = 1.0*l[28]+1.0*l[37]+1.0*l[46]
  out[58] = 1.0*l[29]+1.0*l[38]+1.0*l[47]
  out[59] = 1.0*l[30]
  out[60] = 1.0*l[31]
  out[61] = 1.0*l[32]
  out[62] = 1.0*l[33]+1.0*l[42]+1.0*l[51]
  out[63] = 1.0*l[34]+1.0*l[43]+1.0*l[52]
  out[64] = 1.0*l[39]
  out[65] = 1.0*l[40]
  out[66] = 1.0*l[41]
  out[67] = 1.0*l[48]
  out[68] = 1.0*l[49]
  out[69] = 1.0*l[50]
  out[70] = 1.0*l[55]+1.0*l[56]+1.0*l[64]+1.0*l[65]
  out[71] = 1.0*l[57]+1.0*l[58]+1.0*l[59]
  out[72] = 1.0*l[60]+1.0*l[61]+1.0*l[69]+1.0*l[70]
  out[73] = 1.0*l[66]+1.0*l[67]+1.0*l[68]
  out[74] = 1.0*l[74]+1.0*l[75]+1.0*l[76]+1.0*l[77]+1.0*l[78]

  # pack vert-pairs
  l = np.ravel(tile[:,:,3])
  out[75] = 1.0*l[ 2]+1.0*l[ 3]+1.0*l[ 4]+1.0*l[ 5]+1.0*l[6]
  out[76] = 1.0*l[10]+1.0*l[11]+1.0*l[19]+1.0*l[20]
  out[77] = 1.0*l[12]+1.0*l[13]+1.0*l[14]
  out[78] = 1.0*l[15]+1.0*l[16]+1.0*l[24]+1.0*l[25]
  out[79] = 1.0*l[18]+1.0*l[27]+1.0*l[36]+1.0*l[45]+1.0*l[54]
  out[80] = 1.0*l[21]+1.0*l[22]+1.0*l[23]
  out[81] = 1.0*l[26]+1.0*l[35]+1.0*l[44]+1.0*l[53]+1.0*l[62]
  out[82] = 1.0*l[28]+1.0*l[37]+1.0*l[46]
  out[83] = 1.0*l[29]+1.0*l[38]+1.0*l[47]
  out[84] = 1.0*l[30]
  out[85] = 1.0*l[31]
  out[86] = 1.0*l[32]
  out[87] = 1.0*l[33]+1.0*l[42]+1.0*l[51]
  out[88] = 1.0*l[34]+1.0*l[43]+1.0*l[52]
  out[89] = 1.0*l[39]
  out[90] = 1.0*l[40]
  out[91] = 1.0*l[41]
  out[92] = 1.0*l[48]
  out[93] = 1.0*l[49]
  out[94] = 1.0*l[50]
  out[95] = 1.0*l[55]+1.0*l[56]+1.0*l[64]+1.0*l[65]
  out[96] = 1.0*l[57]+1.0*l[58]+1.0*l[59]
  out[97] = 1.0*l[60]+1.0*l[61]+1.0*l[69]+1.0*l[70]
  out[98] = 1.0*l[66]+1.0*l[67]+1.0*l[68]
  out[99] = 1.0*l[74]+1.0*l[75]+1.0*l[76]+1.0*l[77]+1.0*l[78]

  return out

# pack single
def pack_tile(tile):
  return pack_tile_type1(tile)

# pack all tiles
def pack(tiles):
  output = np.array([[pack_tile(tiles[i,j]) for j in range(tiles.shape[1])] for i in range(tiles.shape[0])])
  return output


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
