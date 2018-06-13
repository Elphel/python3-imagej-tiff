#!/usr/bin/env python3

import numpy as np
import itertools

import pack_tile as pile

# 3,3 of (9,9,2) tiles
A = np.zeros((324,242,9,9,5))
print(A.shape)
d = A.shape

count = 0
for i,j,k,l,m in itertools.product(range(d[0]),range(d[1]),range(d[2]),range(d[3]),range(d[4])):
  A[i,j,k,l,m] = count
  count += 1
  
#B = [[[i*5*5+j*5+k for k in range(5)] for j in range(5)] for i in range(5)]
#B = np.array(B)
#print(B.shape)

def pack_tile(tile):
  return tile.flatten()

B = np.array([[pack_tile(A[i,j]) for j in range(A.shape[1])] for i in range(A.shape[0])])

print(B.shape)