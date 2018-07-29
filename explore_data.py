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

class ExploreData:

    def getComboList(self, top_dir):
        patt = "*-DSI_COMBO.tiff"
        tlist = []
        for i in range(5):
            pp = top_dir#) ,'**', patt) # works
            for j in range (i):
                pp = os.path.join(pp,'*')
            pp = os.path.join(pp,patt)
            tlist += glob.glob(pp)
            print (pp+" "+str(len(tlist)))
        if (self.debug_level > 0):    
            print("Found "+str(len(tlist))+" combo DSI files in "+top_dir+" :")
            if (self.debug_level > 1):    
                print("\n".join(tlist))
        return tlist
    
    def loadComboFiles(self, tlist):
        indx = 0
        images = []
        if (self.debug_level>2):
            print(str(resource.getrusage(resource.RUSAGE_SELF)))
        for combo_file in tlist:
            tiff = ijt.imagej_tiff(combo_file,['disparity_rig','strength_rig'])
            if not indx:
                images = np.empty((len(tlist), tiff.image.shape[0],tiff.image.shape[1],tiff.image.shape[2]), tiff.image.dtype)
            images[indx] = tiff.image
            if (self.debug_level>2):
                print(str(indx)+": "+str(resource.getrusage(resource.RUSAGE_SELF)))
            indx += 1
        return images
    
    def getHistogramDSI(
            self, 
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
            normalize =           True,
            no_histogram =        False            
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
            
#            if not self.good_tiles is None:
#                good_tiles[ids] &= self.good_tiles 
            
            disparity = np.nan_to_num(disparity, copy = False) # to be able to multiply by 0.0 in mask | copy=False, then out=disparity all done in-place
            strength =  np.nan_to_num(strength, copy = False)  # likely should never happen
            np.clip(disparity, disparity_min_clip, disparity_max_clip, out = disparity)
            np.clip(strength, strength_min_clip, strength_max_clip, out = strength)
        if no_histogram:
            strength *= good_tiles[ids]
            return None # no histogram, just condition data
        hist, xedges, yedges = np.histogram2d( # xedges, yedges - just for debugging
            x =      combo_rds[...,1].flatten(),
            y =      combo_rds[...,0].flatten(),
            bins=    (strength_bins, disparity_bins),
            range=   ((strength_min_clip,strength_max_clip),(disparity_min_clip,disparity_max_clip)),
            normed=  normalize,
            weights= good_tiles.flatten())
        return hist, xedges, yedges
    def __init__(self,
               topdir_all,
               debug_level =          0,
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
               hist_sigma =           2.0,  # Blur log histogram
               hist_cutoff=           0.001 #  of maximal  
               ):
    # file name
        self.debug_level = debug_level
        self.disparity_bins =    disparity_bins
        self.strength_bins = strength_bins
        self.disparity_min_drop =  disparity_min_drop
        self.disparity_min_clip = disparity_min_clip
        self.disparity_max_drop = disparity_max_drop
        self.disparity_max_clip = disparity_max_clip
        self.strength_min_drop =  strength_min_drop
        self.strength_min_clip =  strength_min_clip
        self.strength_max_drop =  strength_max_drop
        self.strength_max_clip =  strength_max_clip
        self.hist_sigma =         hist_sigma # Blur log histogram
        self.hist_cutoff=         hist_cutoff #  of maximal  
        self.pre_log_offs =       0.001 # of histogram maximum
        self.good_tiles =         None
        filelist =        self.getComboList(topdir_all)
        combo_rds =            self.loadComboFiles(filelist)
        self.hist, xedges, yedges = self.getHistogramDSI(
                combo_rds =         combo_rds,
                disparity_bins =     self.disparity_bins,
                strength_bins =      self.strength_bins,
                disparity_min_drop = self.disparity_min_drop,
                disparity_min_clip = self.disparity_min_clip,
                disparity_max_drop = self.disparity_max_drop,
                disparity_max_clip = self.disparity_max_clip,
                strength_min_drop =  self.strength_min_drop,
                strength_min_clip =  self.strength_min_clip,
                strength_max_drop =  self.strength_max_drop,
                strength_max_clip =  self.strength_max_clip,
                normalize =          True,
                no_histogram =       False
                
           )
        log_offset = self.pre_log_offs * self.hist.max()
        h_cutoff =   hist_cutoff * self.hist.max()
        lhist =         np.log(self.hist + log_offset)
        blurred_lhist = gaussian_filter(lhist, sigma = self.hist_sigma)
        self.blurred_hist  = np.exp(blurred_lhist) - log_offset
        self.good_tiles =  self.blurred_hist >= h_cutoff
        self.blurred_hist *= self.good_tiles # set bad ones to zero 

    def getTrainDS(self, topdir_train):
        self.trainlist =   self.getComboList(topdir_train)
        self.train_ds =    self.loadComboFiles(self.trainlist)
        self.getHistogramDSI(
                combo_rds =          self.train_ds, # will condition strength
                disparity_bins =     self.disparity_bins,
                strength_bins =      self.strength_bins,
                disparity_min_drop = self.disparity_min_drop,
                disparity_min_clip = self.disparity_min_clip,
                disparity_max_drop = self.disparity_max_drop,
                disparity_max_clip = self.disparity_max_clip,
                strength_min_drop =  self.strength_min_drop,
                strength_min_clip =  self.strength_min_clip,
                strength_max_drop =  self.strength_max_drop,
                strength_max_clip =  self.strength_max_clip,
                normalize =          True,
                no_histogram =       True)

        pass
    def assignBatchBins(self, disp_bins, str_bins):
        hist_to_batch =       np.zeros((self.blurred_hist.shape[0],self.blurred_hist.shape[1]),dtype=int) #zeros_like?
        hist_to_batch_multi = np.ones((self.blurred_hist.shape[0],self.blurred_hist.shape[1]),dtype=int) #zeros_like?
        scale_hist= (disp_bins * str_bins)/self.blurred_hist.sum()
        norm_b_hist =     self.blurred_hist * scale_hist
        disp_list = [] # last disparity hist 
#        disp_multi = [] # number of disp rows to fit
        disp_run_tot = 0.0
        disp_batch = 0
        disp=0
        disp_hist = np.linspace(0,disp_bins * str_bins,disp_bins+1)
        num_batch_bins = disp_bins * str_bins
        batch_index = 0
        num_members = np.zeros((num_batch_bins,),int)
        while disp_batch < disp_bins:
            #disp_multi.append(1)
#        while (disp < self.disparity_bins):
#            disp_target_tot =disp_hist[disp_batch+1]
            disp_run_tot_new = disp_run_tot
            disp0 = disp # start disaprity matching disp_run_tot 
            while (disp_run_tot_new < disp_hist[disp_batch+1]) and (disp < self.disparity_bins):
                disp_run_tot_new += norm_b_hist[:,disp].sum()
                disp+=1;
                disp_multi = 1
                while   (disp_batch < (disp_bins - 1)) and (disp_run_tot_new >= disp_hist[disp_batch+2]):
                    disp_batch += 1 # only if large disp_bins and very high hist value
                    disp_multi += 1
            # now  disp_run_tot - before this batch disparity col
            str_bins_corr = str_bins * disp_multi # if too narrow disparity column - multiply number of strength columns
            str_bins_corr_last = str_bins_corr -1
            str_hist = np.linspace(disp_run_tot, disp_run_tot_new, str_bins_corr + 1)
            str_run_tot_new = disp_run_tot
#            str_batch = 0
            str_index=0
#            wide_col = norm_b_hist[:,disp0:disp] #disp0 - first column, disp - last+ 1
            #iterate in linescan along the column
            for si in range(self.strength_bins):
                for di in range(disp0, disp,1):
                    if norm_b_hist[si,di] > 0.0 :
                        str_run_tot_new += norm_b_hist[si,di]
                        # do not increment after last to avoid precision issues 
                        if (batch_index < num_batch_bins) and (num_members[batch_index] > 0) and (str_index < str_bins_corr_last) and (str_run_tot_new > str_hist[str_index+1]):
                            batch_index += 1
                            str_index +=   1
                        if batch_index < num_batch_bins :     
                            hist_to_batch[si,di] = batch_index
                            num_members[batch_index] += 1
                        else:
                            pass
                    else:
                        hist_to_batch[si,di] = -1
                        
            batch_index += 1 # it was not incremented afterthe last in the column to avoid rounding error 
            disp_batch += 1
            disp_run_tot = disp_run_tot_new
            pass
        return hist_to_batch        
        
        
        
    # total number of layers in tiff

#MAIN
if __name__ == "__main__":
  try:
      topdir_all = sys.argv[1]
  except IndexError:
      topdir_all = "/mnt/dde6f983-d149-435e-b4a2-88749245cc6c/home/eyesis/x3d_data/data_sets/all"#test" #all/"
  try:
      topdir_train = sys.argv[2]
  except IndexError:
      topdir_train = "/mnt/dde6f983-d149-435e-b4a2-88749245cc6c/home/eyesis/x3d_data/data_sets/train"#test" #all/"
      
  ex_data = ExploreData(
               topdir_all =         topdir_all,
               debug_level =          3,
               disparity_bins =     200, #1000,
               strength_bins =      100,
               disparity_min_drop =  -0.1,
               disparity_min_clip =  -0.1,
               disparity_max_drop = 20.0, #100.0,
               disparity_max_clip = 20.0, #100.0,
               strength_min_drop =    0.1,
               strength_min_clip =    0.1,
               strength_max_drop =    1.0,
               strength_max_clip =    0.9,
               hist_sigma =           2.0,  # Blur log histogram
               hist_cutoff=           0.001) #  of maximal  
  """    
  tlist = getComboList(top_dir)
  pre_log_offs =        0.001
  combo_rds= loadComboFiles(tlist)
  hist, xedges, yedges = getHistogramDSI(combo_rds)
  lhist = np.log(hist + pre_log_offs)
  

  blurred = gaussian_filter(lhist, sigma=2.0)
  """
  
  mytitle = "Disparity_Strength histogram"
  fig = plt.figure()
  fig.canvas.set_window_title(mytitle)
  fig.suptitle(mytitle)
#  plt.imshow(lhist,vmin=-6,vmax=-2) # , vmin=0, vmax=.01)
  plt.imshow(ex_data.blurred_hist, vmin=0, vmax=.1 * ex_data.blurred_hist.max())#,vmin=-6,vmax=-2) # , vmin=0, vmax=.01)
  plt.colorbar(orientation='horizontal') # location='bottom')
  bb = ex_data.assignBatchBins(
      disp_bins = 20,
      str_bins=10)
  
  bb_display = bb.copy()
  """
  bb_display %= 20
  bb_mask = (bb > 0)
  bb_scale = 1.0 * (bb_mask)
  bb_scale = (1.05 * bb_scale)
  bb_display = bb_display.astype(float) * bb_scale
  """
#  bb_display = (  ((2 * (bb_display % 2) -1) * (2 * ((bb_display % 20)//10) -1) + 2)/2)*(bb > 0) #).astype(float) 
  bb_display = ( 1+ (bb_display % 2) + 2 * ((bb_display % 20)//10)) * (bb > 0) #).astype(float) 
  
  
  
    
  fig2 = plt.figure()
  fig2.canvas.set_window_title("Batch indices")
  fig2.suptitle("Batch index for each disparity/strength cell")
#  plt.imshow(lhist,vmin=-6,vmax=-2) # , vmin=0, vmax=.01)
  plt.imshow(bb_display) #, vmin=0, vmax=.1 * ex_data.blurred_hist.max())#,vmin=-6,vmax=-2) # , vmin=0, vmax=.01)
  plt.colorbar(orientation='horizontal') # location='bottom')
  
  # Get image stats for train data only
  plt.ioff()
  plt.show()
  ex_data.getTrainDS(topdir_train)
  pass
    
