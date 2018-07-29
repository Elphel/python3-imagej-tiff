#!/usr/bin/env python3
from numpy import float64

__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "andrey@elphel.com"

import os
import sys
import glob
import imagej_tiff as ijt
import numpy as np
import resource
import timeit
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


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
def print_time():
  print(bcolors.BOLDWHITE+"time: "+str(time.time())+bcolors.ENDC)

def getComboList(top_dir):
    patt = "*-DSI_COMBO.tiff"
    tlist = []
    for i in range(5):
        pp = top_dir#) ,'**', patt) # works
        for j in range (i):
            pp = os.path.join(pp,'*')
        pp = os.path.join(pp,patt)
        tlist += glob.glob(pp)
        print (pp+" "+str(len(tlist)))
    print("Found "+str(len(tlist))+" preprocessed tiff files:")
    print("\n".join(tlist))
    return tlist
def loadComboFiles(tlist):
    indx = 0
    images = []
    print(str(resource.getrusage(resource.RUSAGE_SELF)))
    for combo_file in tlist:
        tiff = ijt.imagej_tiff(combo_file,['disparity_rig','strength_rig'])
        if not indx:
            images = np.empty((len(tlist), tiff.image.shape[0],tiff.image.shape[1],tiff.image.shape[2]), tiff.image.dtype)
        images[indx] = tiff.image
        print(str(indx)+": "+str(resource.getrusage(resource.RUSAGE_SELF)))
        indx += 1
    return images

def getHistogramDSI(
        combo_rds,
        disparity_bins =    1000,
        strength_bins =      100,
        disparity_min_drop =  -0.1,
        disparity_min_clip =  -0.1,
        disparity_max_drop = 100.0,
        disparity_max_clip = 100.0,
        strength_min_drop =    0.1,
        strength_min_clip =    0.1,
        strength_max_drop =    1.0,
        strength_max_clip =    0.9,
        normalize =           True
        ):
    good_tiles = np.empty((combo_rds.shape[0], combo_rds.shape[1],combo_rds.shape[2]), dtype=bool)
    for ids in range (combo_rds.shape[0]): #iterate over all scenes ds[2][rows][cols]
        ds = combo_rds[ids]
        disparity = ds[...,0]
        strength =  ds[...,1]
        good_tiles[ids] =  disparity >= disparity_min_drop
        good_tiles[ids] &= disparity <= disparity_max_drop
        good_tiles[ids] &= strength >=  strength_min_drop
        good_tiles[ids] &= strength <=  strength_max_drop
        disparity = np.nan_to_num(disparity, copy = False) # to be able to multiply by 0.0 in mask | copy=False, then out=disparity all done in-place
        strength =  np.nan_to_num(strength, copy = False)  # likely should never happen
        np.clip(disparity, disparity_min_clip, disparity_max_clip, out = disparity)
        np.clip(strength, strength_min_clip, strength_max_clip, out = strength)
        pass
    hist, xedges, yedges = np.histogram2d( # xedges, yedges - just for debugging
        x = combo_rds[...,0].flatten(),
        y = combo_rds[...,1].flatten(),
        bins=(disparity_bins, strength_bins),
        range=((disparity_min_clip,disparity_max_clip),(strength_min_clip,strength_max_clip)),
        normed=normalize,
        weights=good_tiles.flatten())
    return hist, xedges, yedges

#MAIN
if __name__ == "__main__":
  try:
    top_dir = sys.argv[1]
  except IndexError:
    top_dir = "/mnt/dde6f983-d149-435e-b4a2-88749245cc6c/home/eyesis/x3d_data/data_sets/all"#test" #all/"
  tlist = getComboList(top_dir)
  pre_log_offs =        0.001
  combo_rds= loadComboFiles(tlist)
  hist, xedges, yedges = getHistogramDSI(combo_rds)
  lhist = np.log(hist.transpose() + pre_log_offs)
  

  blurred = gaussian_filter(lhist, sigma=2.0)
  
  
  mytitle = "Disparity/Strength histogram"
  fig = plt.figure()
  fig.canvas.set_window_title("Window title: "+mytitle)
  fig.suptitle("Window subtitle: "+mytitle)
#  plt.imshow(lhist,vmin=-6,vmax=-2) # , vmin=0, vmax=.01)
  plt.imshow(blurred)#,vmin=-6,vmax=-2) # , vmin=0, vmax=.01)
  plt.colorbar()
  plt.ioff()
  plt.show()
  pass
    
