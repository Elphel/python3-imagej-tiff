#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math

def hamming_window(x,N):
  y = 0.2 - 0.46*math.cos(2*math.pi*x/(N-1))
  return y

# input: np.array(a,b) - 1 channel
# output: np.array(a,b,3) - 3 color channels
def coldmap(img,zero_span=0.2):

  out = np.dstack(3*[img])

  img_min = np.nanmin(img)
  img_max = np.nanmax(img)

  #print("min: "+str(img_min)+", max: "+str(img_max))

  ch_r = out[...,0]
  ch_g = out[...,1]
  ch_b = out[...,2]

  # blue for <0
  ch_r[img<0] = 0
  ch_g[img<0] = 0
  ch_b[img<0] = -ch_b[img<0]

  # red for >0
  ch_r[img>0] = ch_b[img>0]
  ch_g[img>0] = 0
  ch_b[img>0] = 0

  # green for 0
  ch_r[img==0] = 0
  ch_g[img==0] = img_max
  ch_b[img==0] = 0

  # green for zero vicinity
  ch_r[abs(img)<zero_span/2] = 0
  ch_g[abs(img)<zero_span/2] = img_max/2
  ch_b[abs(img)<zero_span/2] = 0

  return out

# has to be pre transposed
# it just suppose to match
def tiles(img,shape,tiles_per_line=1,borders=True):

  # shape is (n0,n1,n2,n3)
  # n0*n1*n2*n3 = img.shape[1]
  img_min = np.nanmin(img)
  img_max = np.nanmax(img)

  outer_color = [img_max,img_max,img_min]
  outer_color = [img_max,img_max,img_max]

  inner_color = [img_max/4,img_max/4,img_min]
  inner_color = [img_min,img_min,img_min]
  inner_color = [img_max,img_max,img_min]

  group_h = shape[0]
  group_w = shape[1]
  group_size = group_h*group_w

  tile_h = shape[2]
  tile_w = shape[3]
  tile_size = tile_h*tile_w

  tpl = tiles_per_line

  # main

  tmp1 = []

  for i in range(img.shape[0]):

    if i%tpl==0:
      tmp2 = []

    tmp3 = []

    for igh in range(group_h):

      tmp4 = []

      for igw in range(group_w):

        si = (group_w*igh + igw + 0)*tile_size
        ei = (group_w*igh + igw + 1)*tile_size

        tile = img[i,si:ei]
        tile = np.reshape(tile,(tile_h,tile_w,tile.shape[1]))

        if borders:

          if igw==group_w-1:

            b_h_inner = [[inner_color]*(tile_w+0)]*(       1)
            b_h_outer = [[outer_color]*(tile_w+0)]*(       1)
            b_v_outer = [[outer_color]*(       1)]*(tile_h+1)

            # outer hor
            if igh==group_h-1:
              tile = np.concatenate([tile,b_h_outer],axis=0)
            # inner hor
            else:
              tile = np.concatenate([tile,b_h_inner],axis=0)
            # outer vert
            tile = np.concatenate([tile,b_v_outer],axis=1)

          else:

            b_v_inner = [[inner_color]*(       1)]*(tile_h+0)
            b_h_inner = [[inner_color]*(tile_w+1)]*(       1)
            b_h_outer = [[outer_color]*(tile_w+1)]*(       1)

            # inner vert
            tile = np.concatenate([tile,b_v_inner],axis=1)

            # outer hor
            if igh==group_h-1:
              tile = np.concatenate([tile,b_h_outer],axis=0)
            # inner hor
            else:
              tile = np.concatenate([tile,b_h_inner],axis=0)

        tmp4.append(tile)

      tmp3.append(np.concatenate(tmp4,axis=1))

    tmp2.append(np.concatenate(tmp3,axis=0))

    if i%tpl==(tpl-1):
      tmp1.append(np.concatenate(tmp2,axis=1))

  out = np.concatenate(tmp1,axis=0)
  #out = img
  return out

if __name__=="__main__":
  #
  hw = hamming_window
  #
  image = np.array([[1*hw(i,512)*hw(j,512) for i in range(512)] for j in range(512)],np.float32)
  zeros = np.zeros((512,512))

  # 32x324

  #image2 = np.zeros((32,324))
  #rgb_img_0 = tiles(image2,(1,4,9,9),tiles_per_line=2,borders=True)

  #image2 = np.zeros((32,144))
  image2 = np.array([[1*hw(i,144)*hw(j,32) for i in range(144)] for j in range(32)],np.float32)
  #image3 = coldmap(image2)
  rgb_img_0 = tiles(coldmap(image2),(3,3,4,4),tiles_per_line=8,borders=True)

  fig = plt.figure()
  fig.suptitle("HaWi")
  plt.imshow(rgb_img_0)

  rgb_img = coldmap(image)

  #print(rgb_img)

  '''
  for i in range(512):
    for j in range(512):
      if image[i,j]<0:
        rgb_img[i,j,0] = 0
        rgb_img[i,j,1] = 0
        #rgb_img[i,j,2] = 255

      if image[i,j]>0:
        #rgb_img[i,j,0] = 255
        rgb_img[i,j,1] = 0
        rgb_img[i,j,2] = 0

      if image[i,j]==0:
        rgb_img[i,j,0] = 0
        rgb_img[i,j,1] = 255
        rgb_img[i,j,2] = 0
  '''
  print(rgb_img.shape)

  fig = plt.figure()
  fig.suptitle("HamWindow")
  plt.imshow(rgb_img)
  #plt.colorbar()
  plt.show()