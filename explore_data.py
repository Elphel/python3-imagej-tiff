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
import time
import tensorflow as tf

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
TIME_START = time.time()
TIME_LAST  = TIME_START
    
def print_time(txt="",end="\n"):
    global TIME_LAST
    t = time.time()
    if txt:
        txt +=" "
    print(("%s"+bcolors.BOLDWHITE+"at %.4fs (+%.4fs)"+bcolors.ENDC)%(txt,t-TIME_START,t-TIME_LAST), end = end)
    TIME_LAST = t

def _dtype_feature(ndarray):
    """match appropriate tf.train.Feature class with dtype of ndarray. """
    assert isinstance(ndarray, np.ndarray)
    dtype_ = ndarray.dtype
    if dtype_ == np.float64 or dtype_ == np.float32:
        return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
    elif dtype_ == np.int64:
        return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
    else:  
        raise ValueError("The input should be numpy ndarray. \
                           Instead got {}".format(ndarray.dtype))
def readTFRewcordsEpoch(train_filename):
#    filenames = [train_filename]
#    dataset = tf.data.TFRecordDataset(filenames)
    if not  '.tfrecords' in train_filename:
        train_filename += '.tfrecords'
    record_iterator = tf.python_io.tf_record_iterator(path=train_filename)
    corr2d_list=[]
    target_disparity_list=[]
    gt_ds_list = []
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        corr2d_list.append(np.array(example.features.feature['corr2d'] .float_list .value))
        target_disparity_list.append(np.array(example.features.feature['target_disparity'] .float_list .value[0]))
        gt_ds_list.append(np.array(example.features.feature['gt_ds'] .float_list .value))
    corr2d=            np.array(corr2d_list)
    target_disparity = np.array(target_disparity_list)
    gt_ds =            np.array(gt_ds_list)
    return corr2d, target_disparity, gt_ds   


class ExploreData:
    PATTERN = "*-DSI_COMBO.tiff"
#    ML_DIR = "ml"
    ML_PATTERN = "*-ML_DATA*.tiff"
    """
1527182801_296892-ML_DATARND-32B-O-FZ0.05-OFFS-0.20000_0.20000.tiff
    """
    def getComboList(self, top_dir):
#        patt = "*-DSI_COMBO.tiff"
        tlist = []
        for i in range(5):
            pp = top_dir#) ,'**', patt) # works
            for j in range (i):
                pp = os.path.join(pp,'*')
            pp = os.path.join(pp, ExploreData.PATTERN)
            tlist += glob.glob(pp)
            if (self.debug_level > 0):    
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
            list_rds,
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
        good_tiles_list=[]
        for combo_rds in list_rds:
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
#                if no_histogram:
#                    strength *= good_tiles[ids]
#            if no_histogram:
#                return None # no histogram, just condition data
            good_tiles_list.append(good_tiles)
        combo_rds = np.concatenate(list_rds)
        hist, xedges, yedges = np.histogram2d( # xedges, yedges - just for debugging
            x =      combo_rds[...,1].flatten(),
            y =      combo_rds[...,0].flatten(),
            bins=    (strength_bins, disparity_bins),
            range=   ((strength_min_clip,strength_max_clip),(disparity_min_clip,disparity_max_clip)),
            normed=  normalize,
            weights= np.concatenate(good_tiles_list).flatten())
        for i, combo_rds in enumerate(list_rds):
            for ids in range (combo_rds.shape[0]): #iterate over all scenes ds[2][rows][cols]
#                strength =  combo_rds[ids][...,1]
#                strength *= good_tiles_list[i][ids]
                combo_rds[ids][...,1]*= good_tiles_list[i][ids]
        return hist, xedges, yedges
    
    
    def __init__(self,
               topdir_train,
               topdir_test,
               ml_subdir,  
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
        #self.testImageTiles()    
        self.disparity_bins =     disparity_bins
        self.strength_bins =      strength_bins
        self.disparity_min_drop = disparity_min_drop
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
        self.files_train =        self.getComboList(topdir_train)
        self.files_test =         self.getComboList(topdir_test)
        
        self.train_ds =           self.loadComboFiles(self.files_train)
        self.test_ds =            self.loadComboFiles(self.files_test)
        
        self.num_tiles = self.train_ds.shape[1]*self.train_ds.shape[2] 
        self.hist, xedges, yedges = self.getHistogramDSI(
                list_rds =           [self.train_ds,self.test_ds], # combo_rds,
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

    def exploreNeibs(self,
                     data_ds, # disparity/strength data for all files (train or test)
                     radius,  # how far to look from center each side ( 1- 3x3, 2 - 5x5)
                     disp_thesh = 5.0): # reduce effective variance for higher disparities
        """
        For each tile calculate difference between max and min among neighbors and number of qualifying neighbors (bad cewnter is not removed)
        """
        disp_min =   np.empty_like(data_ds[...,0], dtype = np.float)
        disp_max =   np.empty_like(disp_min, dtype = np.float)
        tile_neibs = np.zeros_like(disp_min, dtype = np.int)
        dmin = data_ds[...,0].min()
        dmax = data_ds[...,0].max()
        good_tiles = self.getBB(data_ds) >= 0
        side = 2 * radius + 1
        for nf, ds in enumerate(data_ds):
            disp = ds[...,0] 
            height = disp.shape[0]
            width = disp.shape[1]
            bad_max = np.ones((height+side, width+side),  dtype=float) * dmax
            bad_min = np.ones((height+side, width+side),  dtype=float) * dmin
            good =    np.zeros((height+side, width+side), dtype=int)
            #Assign centers of the array, replace bad tiles with max/min (so they will not change min/max) 
            bad_max[radius:height+radius,radius:width+radius] = np.select([good_tiles[nf]],[disp],default = dmax)
            bad_min[radius:height+radius,radius:width+radius] = np.select([good_tiles[nf]],[disp],default = dmin)
            good   [radius:height+radius,radius:width+radius] = good_tiles[nf]
            disp_min  [nf,...] = disp 
            disp_max  [nf,...] = disp
            tile_neibs[nf,...] = good_tiles[nf]
            for offset_y in range(-radius, radius+1):
                oy = offset_y+radius
                for offset_x in range(-radius, radius+1):
                    ox = offset_x+radius
                    if offset_y or offset_x: # Skip center - already copied
                        np.minimum(disp_min[nf], bad_max[oy:oy+height, ox:ox+width], out=disp_min[nf])
                        np.maximum(disp_max[nf], bad_min[oy:oy+height, ox:ox+width], out=disp_max[nf])
                        tile_neibs[nf] +=  good[oy:oy+height, ox:ox+width]
                        pass
                    pass
                pass
            pass
        
        #disp_thesh
        disp_avar = disp_max - disp_min
        disp_rvar = disp_avar * disp_thesh / np.maximum(disp_max, 0.001) # removing division by 0 error - those tiles will be anyway discarded 
        disp_var = np.select([disp_max >= disp_thesh, disp_max < disp_thesh],[disp_rvar,disp_avar])
        return disp_var, tile_neibs

    def assignBatchBins(self,
                        disp_bins,
                        str_bins,
                        files_per_scene = 5,   # not used here, will be used when generating batches
                        min_batch_choices=10,  # not used here, will be used when generating batches
                        max_batch_files = 10): # not used here, will be used when generating batches
        """
        for each disparity/strength combination (self.disparity_bins * self.strength_bins = 1000*100) provide number of "large"
        variable-size disparity/strength bin, or -1 if this disparity/strength combination does not seem right
        """
        self.files_per_scene = files_per_scene
        self.min_batch_choices=min_batch_choices
        self.max_batch_files = max_batch_files
        
        hist_to_batch =       np.zeros((self.blurred_hist.shape[0],self.blurred_hist.shape[1]),dtype=int) #zeros_like?
        hist_to_batch_multi = np.ones((self.blurred_hist.shape[0],self.blurred_hist.shape[1]),dtype=int) #zeros_like?
        scale_hist= (disp_bins * str_bins)/self.blurred_hist.sum()
        norm_b_hist =     self.blurred_hist * scale_hist
        disp_list = [] # last disparity hist 
#        disp_multi = [] # number of disp rows to fit
        disp_run_tot = 0.0
        disp_batch = 0
        disp=0
        num_batch_bins = disp_bins * str_bins
        disp_hist = np.linspace(0, num_batch_bins, disp_bins+1)
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
        self.hist_to_batch = hist_to_batch
        return hist_to_batch        

    def getBB(self, data_ds):
        """
        for each file, each tile get histogram index (or -1 for bad tiles)
        """
        hist_to_batch = self.hist_to_batch
        files_batch_list = []
        disp_step = ( self.disparity_max_clip - self.disparity_min_clip )/ self.disparity_bins 
        str_step =  ( self.strength_max_clip -  self.strength_min_clip )/ self.strength_bins
        bb = np.empty_like(data_ds[...,0],dtype=int)
        for findx in range(data_ds.shape[0]):
            ds = data_ds[findx]
            gt = ds[...,1] > 0.0 # OK
            db = (((ds[...,0] - self.disparity_min_clip)/disp_step).astype(int))*gt
            sb = (((ds[...,1] - self.strength_min_clip)/ str_step).astype(int))*gt
            np.clip(db, 0, self.disparity_bins-1, out = db)
            np.clip(sb, 0, self.strength_bins-1, out = sb)
            bb[findx] = (self.hist_to_batch[sb.reshape(self.num_tiles),db.reshape(self.num_tiles)])   .reshape(db.shape[0],db.shape[1]) + (gt -1)
        return bb

    def makeBatchLists(self,
            data_ds =      None, # (disparity,strength) per scene, per tile
            disp_var =     None, # difference between maximal and minimal disparity for each scene, each tile
            disp_neibs =   None, # number of valid tiles around each center tile (for 3x3 (radius = 1) - macximal is 9  
            min_var =      None, # Minimal tile variance to include
            max_var =      None, # Maximal tile variance to include
            min_neibs =    None):# Minimal number of valid tiles to include
        if data_ds is None:
             data_ds =      self.train_ds
        hist_to_batch = self.hist_to_batch
        num_batch_tiles = np.empty((data_ds.shape[0],self.hist_to_batch.max()+1),dtype = int) 
        bb = self.getBB(data_ds)
        use_neibs = not ((disp_var is None) or (disp_neibs is None) or (min_var is None) or (max_var is None) or (min_neibs is None))
        list_of_file_lists=[]
        for findx in range(data_ds.shape[0]):
            foffs = findx * self.num_tiles 
            lst = []
            for i in range (self.hist_to_batch.max()+1):
                lst.append([])
#            bb1d = bb[findx].reshape(self.num_tiles)
            if use_neibs:    
                disp_var_tiles =   disp_var[findx].reshape(self.num_tiles)
                disp_neibs_tiles = disp_neibs[findx].reshape(self.num_tiles)
            for n, indx in enumerate(bb[findx].reshape(self.num_tiles)):
                if indx >= 0:
                    if use_neibs:
#                        disp_var_tiles =   disp_var[findx].reshape(self.num_tiles)
#                        disp_neibs_tiles = disp_neibs[findx].reshape(self.num_tiles)
                        if disp_neibs_tiles[n] < min_neibs:
                            continue # too few neighbors
                        if not disp_var_tiles[n] >= min_var:
                            continue #too small variance 
                        if not disp_var_tiles[n] <  max_var:
                            continue #too large variance 
                    lst[indx].append(foffs + n)
            lst_arr=[]
            for i,l in enumerate(lst):
#                lst_arr.append(np.array(l,dtype = int))
                lst_arr.append(l)
                num_batch_tiles[findx,i] = len(l)
            list_of_file_lists.append(lst_arr)
        self.list_of_file_lists= list_of_file_lists
        self.num_batch_tiles =   num_batch_tiles
        return list_of_file_lists, num_batch_tiles
    #todo: only use other files if there are no enough choices in the main file!
    
    
    def augmentBatchFileIndices(self,
                                 seed_index,
                                 min_choices=None,
                                 max_files = None,
                                 set_ds = None
                                 ):
        if min_choices is None:
            min_choices = self.min_batch_choices 
        if max_files is None:
            max_files =  self.max_batch_files
        if set_ds is None:
            set_ds = self.train_ds
        full_num_choices = self.num_batch_tiles[seed_index].copy()
        flist = [seed_index]
        all_choices = list(range(self.num_batch_tiles.shape[0]))
        all_choices.remove(seed_index)
        for _ in range (max_files-1):
            if full_num_choices.min() >= min_choices:
                break
            findx = np.random.choice(all_choices)
            flist.append(findx)
            all_choices.remove(findx)
            full_num_choices += self.num_batch_tiles[findx]

        file_tiles_sparse = [[] for _ in set_ds] #list of empty lists for each train scene (will be sparse) 
        for nt in range(self.num_batch_tiles.shape[1]): #number of tiles per batch (not counting ml file variant)
            tl = []
            nchoices = 0
            for findx in flist:
                if (len(self.list_of_file_lists[findx][nt])):
                    tl.append(self.list_of_file_lists[findx][nt])
                nchoices+= self.num_batch_tiles[findx][nt]
                if nchoices >= min_choices: # use minimum of extra files
                    break;
            tile = np.random.choice(np.concatenate(tl))
#            print (nt, tile, tile//self.num_tiles, tile % self.num_tiles)
            if not type (tile) is np.int64:
                print("tile=",tile)
            file_tiles_sparse[tile//self.num_tiles].append(tile % self.num_tiles)
        file_tiles = []
        for findx in flist:
            file_tiles.append(np.sort(np.array(file_tiles_sparse[findx],dtype=int))) 
        return flist, file_tiles # file indices, list if tile indices for each file   
            
            
               
                
    def getMLList(self, ml_subdir, flist):
        ml_list = []
        for fn in flist:
            ml_patt = os.path.join(os.path.dirname(fn), ml_subdir, ExploreData.ML_PATTERN)
            ml_list.append(glob.glob(ml_patt))
##        self.ml_list = ml_list
        return ml_list
            
    def getBatchData(
            self,
            flist,
            tiles,
            ml_list,
            ml_num = None ): # 0 - use all ml files for the scene, >0 select random number
        if ml_num is None:
            ml_num = self.files_per_scene
        ml_all_files = []
        for findx in flist:
            mli =  list(range(len(ml_list[findx])))
            if (ml_num > 0) and (ml_num < len(mli)):
                mli_left = mli
                mli = []
                for _ in range(ml_num):
                    ml = np.random.choice(mli_left)
                    mli.append(ml)
                    mli_left.remove(ml)
            ml_files = []
            for ml_index in mli:
                ml_files.append(ml_list[findx][ml_index])
            ml_all_files.append(ml_files)        
                    
        return ml_all_files
    
    def prepareBatchData(self, ml_list, seed_index, min_choices=None, max_files = None, ml_num = None, set_ds = None, radius = 0):
        if min_choices is None:
            min_choices = self.min_batch_choices
        if max_files is None:
            max_files = self.max_batch_files
        if ml_num is None:
            ml_num = self.files_per_scene
        if set_ds is None:
            set_ds = self.train_ds
        tiles_in_sample = (2 * radius + 1) * (2 * radius + 1)
        height = set_ds.shape[1]
        width =  set_ds.shape[2]
        width_m1 = width-1
        height_m1 = height-1
#        set_ds = [self.train_ds, self.test_ds][test_set]            
        corr_layers =  ['hor-pairs', 'vert-pairs','diagm-pair', 'diago-pair']
        flist,tiles = self.augmentBatchFileIndices(seed_index, min_choices, max_files, set_ds)
        
#        ml_all_files = self.getBatchData(flist, tiles, self.ml_list,  ml_num) # 0 - use all ml files for the scene, >0 select random number
        ml_all_files = self.getBatchData(flist, tiles, ml_list,  ml_num) # 0 - use all ml files for the scene, >0 select random number
        if self.debug_level > 1:
            print ("==============",seed_index, flist)
            for i, findx in enumerate(flist):
                print(i,"\n".join(ml_all_files[i])) 
                print(tiles[i]) 
        total_tiles = 0
        for i, t in enumerate(tiles):
            total_tiles += len(t)*len(ml_all_files[i]) # tiles per scene * offset files per scene
        if self.debug_level > 1:
            print("Tiles in the batch=",total_tiles)
        corr2d_batch = None # np.empty((total_tiles, len(corr_layers),81))
        gt_ds_batch =            np.empty((total_tiles * tiles_in_sample, 2), dtype=float) 
        target_disparity_batch = np.empty((total_tiles * tiles_in_sample, ),  dtype=float) 
        start_tile = 0
        for nscene, scene_files in enumerate(ml_all_files):
            for path in  scene_files:
                '''
                Create tiles list including neighbors
                '''
                full_tiles = np.empty([len(tiles[nscene]) * tiles_in_sample], dtype = int)
                indx = 0;
                for i, nt in enumerate(tiles[nscene]):
                    ty = nt // width
                    tx = nt % width
                    for dy in range (-radius, radius+1):
                        y = np.clip(ty+dy,0,height_m1)
                        for dx in range (-radius, radius+1):
                            x = np.clip(tx+dx,0,width_m1)
                            full_tiles[indx] = y * width + x
                            indx += 1
                #now tile_list is np.array instead of the list, but it seems to be OK
                img = ijt.imagej_tiff(path, corr_layers, tile_list=full_tiles) # tiles[nscene])
                corr2d =           img.corr2d
                target_disparity = img.target_disparity
                gt_ds =            img.gt_ds
                end_tile = start_tile + corr2d.shape[0]
                 
                if corr2d_batch is None:
#                    corr2d_batch = np.empty((total_tiles, tiles_in_sample * len(corr_layers), corr2d.shape[-1]))
                    corr2d_batch = np.empty((total_tiles * tiles_in_sample, len(corr_layers), corr2d.shape[-1]))
                gt_ds_batch            [start_tile:end_tile] = gt_ds
                target_disparity_batch [start_tile:end_tile] = target_disparity
                corr2d_batch           [start_tile:end_tile] = corr2d
                start_tile = end_tile
                """
                 Sometimes get bad tile in ML file that was not bad in COMBO-DSI
                 Need to recover
                 np.argwhere(np.isnan(target_disparity_batch))                 
                """
        bad_tiles = np.argwhere(np.isnan(target_disparity_batch))
        if (len(bad_tiles)>0):
            print ("*** Got %d bad tiles in a batch, replacing..."%(len(bad_tiles)), end=" ")
            # for now - just repeat some good tile
            for ibt in bad_tiles:
                while np.isnan(target_disparity_batch[ibt]):
                    irt = np.random.randint(0,total_tiles)
                    if not np.isnan(target_disparity_batch[irt]):
                        target_disparity_batch[ibt] = target_disparity_batch[irt]
                        corr2d_batch[ibt] = corr2d_batch[irt]
                        gt_ds_batch[ibt] = gt_ds_batch[irt]
                        break
            print (" done replacing")
        self.corr2d_batch =           corr2d_batch
        self.target_disparity_batch = target_disparity_batch
        self.gt_ds_batch =            gt_ds_batch
        return corr2d_batch, target_disparity_batch, gt_ds_batch

    def writeTFRewcordsEpoch(self, tfr_filename, ml_list, files_list = None, set_ds= None, radius = 0): # test_set=False):
#        train_filename = 'train.tfrecords'  # address to save the TFRecords file
        # open the TFRecords file
        if not  '.tfrecords' in tfr_filename:
            tfr_filename += '.tfrecords'

        tfr_filename=tfr_filename.replace(' ','_')
        if files_list is None:
            files_list = self.files_train
            
        if set_ds is None:
            set_ds = self.train_ds
            
        writer = tf.python_io.TFRecordWriter(tfr_filename)
#$        files_list = [self.files_train, self.files_test][test_set]
        seed_list = np.arange(len(files_list))
        np.random.shuffle(seed_list)
        cluster_size = (2 * radius + 1) * (2 * radius + 1)
        for nscene, seed_index in enumerate(seed_list):
            corr2d_batch, target_disparity_batch, gt_ds_batch = ex_data.prepareBatchData(ml_list, seed_index, min_choices=None, max_files = None, ml_num = None, set_ds = set_ds, radius = radius)
            #shuffles tiles in a batch
#            tiles_in_batch = len(target_disparity_batch)
            tiles_in_batch =    corr2d_batch.shape[0]
            clusters_in_batch = tiles_in_batch // cluster_size
#            permut = np.random.permutation(tiles_in_batch)
            permut = np.random.permutation(clusters_in_batch)
            corr2d_clusters =           corr2d_batch.          reshape((clusters_in_batch,-1)) 
            target_disparity_clusters = target_disparity_batch.reshape((clusters_in_batch,-1)) 
            gt_ds_clusters =            gt_ds_batch.           reshape((clusters_in_batch,-1)) 
                
#            corr2d_batch_shuffled =           corr2d_batch[permut].reshape((corr2d_batch.shape[0], corr2d_batch.shape[1]*corr2d_batch.shape[2]))
#            target_disparity_batch_shuffled = target_disparity_batch[permut].reshape((tiles_in_batch,1))
#            gt_ds_batch_shuffled =            gt_ds_batch[permut]

            corr2d_batch_shuffled =           corr2d_clusters[permut].          reshape((tiles_in_batch, -1))
            target_disparity_batch_shuffled = target_disparity_clusters[permut].reshape((tiles_in_batch, -1))
            gt_ds_batch_shuffled =            gt_ds_clusters[permut].           reshape((tiles_in_batch, -1))
            
            if nscene == 0:
                dtype_feature_corr2d =   _dtype_feature(corr2d_batch_shuffled)
                dtype_target_disparity = _dtype_feature(target_disparity_batch_shuffled)
                dtype_feature_gt_ds =    _dtype_feature(gt_ds_batch_shuffled)

            for i in range(tiles_in_batch):
                x = corr2d_batch_shuffled[i].astype(np.float32)
                y = target_disparity_batch_shuffled[i].astype(np.float32)
                z = gt_ds_batch_shuffled[i].astype(np.float32)
                d_feature = {'corr2d':          dtype_feature_corr2d(x),
                             'target_disparity':dtype_target_disparity(y),
                             'gt_ds':           dtype_feature_gt_ds(z)}
                example = tf.train.Example(features=tf.train.Features(feature=d_feature))
                writer.write(example.SerializeToString())
            if (self.debug_level > 0):
                print("Scene %d of %d -> %s"%(nscene, len(seed_list), tfr_filename))        
        writer.close()
        sys.stdout.flush()        
    
    def showVariance(self,
            rds_list,           # list of disparity/strength files, suchas training, testing 
            disp_var_list,      # list of disparity variance files. Same shape(but last dim) as rds_list
            num_neibs_list,    # list of number of tile neibs files. Same shape(but last dim) as rds_list
            variance_min =       0.0,
            variance_max =       1.5,
            neibs_min =          9,
            #Same parameters as for the histogram 
#            disparity_bins =    1000,
#            strength_bins =      100,
#            disparity_min_drop =  -0.1,
#            disparity_min_clip =  -0.1,
#            disparity_max_drop = 100.0,
#            disparity_max_clip = 100.0,
#            strength_min_drop =    0.1,
#            strength_min_clip =    0.1,
#            strength_max_drop =    1.0,
#            strength_max_clip =    0.9,
            normalize =           False): # True):
        good_tiles_list=[]
        for nf, combo_rds in enumerate(rds_list):
            disp_var =  disp_var_list[nf]
            num_neibs = num_neibs_list[nf]
            good_tiles = np.empty((combo_rds.shape[0], combo_rds.shape[1],combo_rds.shape[2]), dtype=bool)
            for ids in range (combo_rds.shape[0]): #iterate over all scenes ds[2][rows][cols]
                ds = combo_rds[ids]
                disparity = ds[...,0]
                strength =  ds[...,1]
                variance =  disp_var[ids]
                neibs =     num_neibs[ids]
                good_tiles[ids] =  disparity >= self.disparity_min_drop
                good_tiles[ids] &= disparity <= self.disparity_max_drop
                good_tiles[ids] &= strength >=  self.strength_min_drop
                good_tiles[ids] &= strength <=  self.strength_max_drop
                good_tiles[ids] &= neibs    >=  neibs_min
                good_tiles[ids] &= variance >=  variance_min
                good_tiles[ids] &= variance <   variance_max
                disparity = np.nan_to_num(disparity, copy = False) # to be able to multiply by 0.0 in mask | copy=False, then out=disparity all done in-place
                strength =  np.nan_to_num(strength, copy = False)  # likely should never happen
#                np.clip(disparity, self.disparity_min_clip, self.disparity_max_clip, out = disparity)
#                np.clip(strength, self.strength_min_clip, self.strength_max_clip, out = strength)
            good_tiles_list.append(good_tiles)
        combo_rds = np.concatenate(rds_list)
        hist, xedges, yedges = np.histogram2d( # xedges, yedges - just for debugging
            x =      combo_rds[...,1].flatten(),
            y =      combo_rds[...,0].flatten(),
            bins=    (self.strength_bins, self.disparity_bins),
            range=   ((self.strength_min_clip,self.strength_max_clip),(self.disparity_min_clip,self.disparity_max_clip)),
            normed=  normalize,
            weights= np.concatenate(good_tiles_list).flatten())
        
        mytitle = "Disparity_Strength variance histogram"
        fig = plt.figure()
        fig.canvas.set_window_title(mytitle)
        fig.suptitle("Min variance = %f, max variance = %f, min neibs = %d"%(variance_min, variance_max, neibs_min))
#        plt.imshow(hist, vmin=0, vmax=.1 * hist.max())#,vmin=-6,vmax=-2) # , vmin=0, vmax=.01)
        plt.imshow(hist, vmin=0.0, vmax=300.0)#,vmin=-6,vmax=-2) # , vmin=0, vmax=.01)
        plt.colorbar(orientation='horizontal') # location='bottom')
        
#        for i, combo_rds in enumerate(rds_list):
#            for ids in range (combo_rds.shape[0]): #iterate over all scenes ds[2][rows][cols]
#                combo_rds[ids][...,1]*= good_tiles_list[i][ids]
#        return hist, xedges, yedges

#MAIN
if __name__ == "__main__":
  try:
      topdir_train = sys.argv[1]
  except IndexError:
      topdir_train = "/mnt/dde6f983-d149-435e-b4a2-88749245cc6c/home/eyesis/x3d_data/data_sets/train"#test" #all/"
  try:
      topdir_test = sys.argv[2]
  except IndexError:
      topdir_test = "/mnt/dde6f983-d149-435e-b4a2-88749245cc6c/home/eyesis/x3d_data/data_sets/test"#test" #all/"
      
  try:
      pathTFR =     sys.argv[3]
  except IndexError:
      pathTFR = "/mnt/dde6f983-d149-435e-b4a2-88749245cc6c/home/eyesis/x3d_data/data_sets/tf_data_3x3a" #no trailing "/"

  try:
      ml_subdir =   sys.argv[4]
  except IndexError:
      ml_subdir =   "ml"

  #Parameters to generate neighbors data. Set radius to 0 to generate single-tile     
  RADIUS = 1
  MIN_NEIBS = (2 * RADIUS + 1) * (2 * RADIUS + 1) # All tiles valid == 9
  VARIANCE_THRESHOLD = 1.5
  NUM_TRAIN_SETS = 8
 
  if RADIUS == 0:
    BATCH_DISP_BINS = 20
    BATCH_STR_BINS =  10
  else:
    BATCH_DISP_BINS = 8
    BATCH_STR_BINS =  3

  train_filenameTFR = pathTFR+"/train"        
  test_filenameTFR =  pathTFR+"/test"
#        disp_bins = 20,
#      str_bins=10)

#  corr2d, target_disparity, gt_ds = readTFRewcordsEpoch(train_filenameTFR)
#  print_time("Read %d tiles"%(corr2d.shape[0]))
#  exit (0)
  ex_data = ExploreData(
               topdir_train =         topdir_train,
               topdir_test =          topdir_test,
               ml_subdir =            ml_subdir,
               debug_level =          1, #3, ##0, #3,
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
  
  mytitle = "Disparity_Strength histogram"
  fig = plt.figure()
  fig.canvas.set_window_title(mytitle)
  fig.suptitle(mytitle)
#  plt.imshow(lhist,vmin=-6,vmax=-2) # , vmin=0, vmax=.01)
  plt.imshow(ex_data.blurred_hist, vmin=0, vmax=.1 * ex_data.blurred_hist.max())#,vmin=-6,vmax=-2) # , vmin=0, vmax=.01)
  plt.colorbar(orientation='horizontal') # location='bottom')
  hist_to_batch = ex_data.assignBatchBins(
      disp_bins = BATCH_DISP_BINS,
      str_bins =  BATCH_STR_BINS)
  bb_display = hist_to_batch.copy()
  bb_display = ( 1+ (bb_display % 2) + 2 * ((bb_display % 20)//10)) * (hist_to_batch > 0) #).astype(float) 
  fig2 = plt.figure()
  fig2.canvas.set_window_title("Batch indices")
  fig2.suptitle("Batch index for each disparity/strength cell")
  plt.imshow(bb_display) #, vmin=0, vmax=.1 * ex_data.blurred_hist.max())#,vmin=-6,vmax=-2) # , vmin=0, vmax=.01)
  
  """ prepare test dataset """
#  RADIUS = 1
#  MIN_NEIBS = (2 * RADIUS + 1) * (2 * RADIUS + 1) # All tiles valid
#  VARIANCE_THRESHOLD = 1.5

  if (RADIUS > 0):
      disp_var_test,  num_neibs_test =  ex_data.exploreNeibs(ex_data.test_ds, RADIUS)
      disp_var_train, num_neibs_train = ex_data.exploreNeibs(ex_data.train_ds, RADIUS)
      
      # show varinace histogram
#      for var_thresh in [0.1, 1.0, 1.5, 2.0, 5.0]:
      for var_thresh in [1.5]:
           ex_data.showVariance(
                rds_list =       [ex_data.train_ds, ex_data.test_ds],           # list of disparity/strength files, suchas training, testing 
                disp_var_list =  [disp_var_train,  disp_var_test],      # list of disparity variance files. Same shape(but last dim) as rds_list
                num_neibs_list = [num_neibs_train, num_neibs_test],    # list of number of tile neibs files. Same shape(but last dim) as rds_list
                variance_min =       0.0,
                variance_max =       var_thresh,
                neibs_min =          9)
           ex_data.showVariance(
                rds_list =       [ex_data.train_ds, ex_data.test_ds],           # list of disparity/strength files, suchas training, testing 
                disp_var_list =  [disp_var_train,  disp_var_test],      # list of disparity variance files. Same shape(but last dim) as rds_list
                num_neibs_list = [num_neibs_train, num_neibs_test],    # list of number of tile neibs files. Same shape(but last dim) as rds_list
                variance_min =       var_thresh,
                variance_max =       1000.0,
                neibs_min =          9)
           pass
      pass
      
  else:
      disp_var_test,  num_neibs_test =  None, None    
      disp_var_train, num_neibs_train = None, None    
  
  ml_list_train=ex_data.getMLList(ml_subdir, ex_data.files_train)
  ml_list_test= ex_data.getMLList(ml_subdir, ex_data.files_test)

  if RADIUS == 0 :
      list_of_file_lists_train, num_batch_tiles_train = ex_data.makeBatchLists( # results are also saved to self.*
          data_ds =      ex_data.train_ds,
          disp_var =     disp_var_train,      # difference between maximal and minimal disparity for each scene, each tile
          disp_neibs =   num_neibs_train,     # number of valid tiles around each center tile (for 3x3 (radius = 1) - macximal is 9  
          min_var =      0.0,                # Minimal tile variance to include
          max_var =      VARIANCE_THRESHOLD, # Maximal tile variance to include
          min_neibs =    MIN_NEIBS)          # Minimal number of valid tiles to include
      pass
#  ex_data.makeBatchLists(data_ds = ex_data.train_ds)
      for train_var in range (NUM_TRAIN_SETS):
          fpath =  train_filenameTFR+("%03d"%(train_var,))
          ex_data.writeTFRewcordsEpoch(fpath, ml_list = ml_list_train, files_list = ex_data.files_train, set_ds= ex_data.train_ds)
          
      list_of_file_lists_test, num_batch_tiles_test = ex_data.makeBatchLists( # results are also saved to self.*
          data_ds =      ex_data.test_ds,
          disp_var =     disp_var_test,      # difference between maximal and minimal disparity for each scene, each tile
          disp_neibs =   num_neibs_test,     # number of valid tiles around each center tile (for 3x3 (radius = 1) - macximal is 9  
          min_var =      0.0,                # Minimal tile variance to include
          max_var =      VARIANCE_THRESHOLD, # Maximal tile variance to include
          min_neibs =    MIN_NEIBS)          # Minimal number of valid tiles to include
      fpath =  test_filenameTFR # +("-%03d"%(train_var,))
      ex_data.writeTFRewcordsEpoch(fpath, ml_list = ml_list_train, files_list = ex_data.files_test, set_ds= ex_data.test_ds)
      pass
  else: # RADIUS > 0
      # train
      list_of_file_lists_train, num_batch_tiles_train = ex_data.makeBatchLists( # results are also saved to self.*
          data_ds =      ex_data.train_ds,
          disp_var =     disp_var_train,      # difference between maximal and minimal disparity for each scene, each tile
          disp_neibs =   num_neibs_train,     # number of valid tiles around each center tile (for 3x3 (radius = 1) - macximal is 9  
          min_var =      0.0,                # Minimal tile variance to include
          max_var =      VARIANCE_THRESHOLD, # Maximal tile variance to include
          min_neibs =    MIN_NEIBS)          # Minimal number of valid tiles to include
      num_le_train = num_batch_tiles_train.sum()
      print("Number of <= %f disparity variance tiles: %d (train)"%(VARIANCE_THRESHOLD, num_le_train))
      for train_var in range (NUM_TRAIN_SETS):
          fpath =  train_filenameTFR+("%03d_R%d_LE%4.1f"%(train_var,RADIUS,VARIANCE_THRESHOLD))
          ex_data.writeTFRewcordsEpoch(fpath, ml_list = ml_list_train, files_list = ex_data.files_train, set_ds= ex_data.train_ds, radius = RADIUS)

      list_of_file_lists_train, num_batch_tiles_train = ex_data.makeBatchLists( # results are also saved to self.*
          data_ds =      ex_data.train_ds,
          disp_var =     disp_var_train,      # difference between maximal and minimal disparity for each scene, each tile
          disp_neibs =   num_neibs_train,     # number of valid tiles around each center tile (for 3x3 (radius = 1) - macximal is 9  
          min_var =      VARIANCE_THRESHOLD,  # Minimal tile variance to include
          max_var =      1000.0,              # Maximal tile variance to include
          min_neibs =    MIN_NEIBS)          # Minimal number of valid tiles to include
      num_gt_train = num_batch_tiles_train.sum()
      high_fract_train = 1.0 * num_gt_train / (num_le_train + num_gt_train)
      print("Number of > %f disparity variance tiles: %d, fraction = %f (train)"%(VARIANCE_THRESHOLD, num_gt_train, high_fract_train))
      for train_var in range (NUM_TRAIN_SETS):
          fpath =  (train_filenameTFR+("%03d_R%d_GT%4.1f"%(train_var,RADIUS,VARIANCE_THRESHOLD)))
          ex_data.writeTFRewcordsEpoch(fpath, ml_list = ml_list_train, files_list = ex_data.files_train, set_ds= ex_data.train_ds, radius = RADIUS)
          
      # test
      list_of_file_lists_test, num_batch_tiles_test = ex_data.makeBatchLists( # results are also saved to self.*
      data_ds =      ex_data.test_ds,
      disp_var =     disp_var_test,      # difference between maximal and minimal disparity for each scene, each tile
      disp_neibs =   num_neibs_test,     # number of valid tiles around each center tile (for 3x3 (radius = 1) - macximal is 9  
      min_var =      0.0,                # Minimal tile variance to include
      max_var =      VARIANCE_THRESHOLD, # Maximal tile variance to include
      min_neibs =    MIN_NEIBS)          # Minimal number of valid tiles to include
      num_le_test = num_batch_tiles_test.sum()
      print("Number of <= %f disparity variance tiles: %d (est)"%(VARIANCE_THRESHOLD, num_le_test))

      fpath =  test_filenameTFR +("TEST_R%d_LE%4.1f"%(RADIUS,VARIANCE_THRESHOLD))
      ex_data.writeTFRewcordsEpoch(fpath, ml_list = ml_list_test, files_list = ex_data.files_test, set_ds= ex_data.test_ds, radius = RADIUS)

      list_of_file_lists_test, num_batch_tiles_test = ex_data.makeBatchLists( # results are also saved to self.*
      data_ds =      ex_data.test_ds,
      disp_var =     disp_var_test,      # difference between maximal and minimal disparity for each scene, each tile
      disp_neibs =   num_neibs_test,     # number of valid tiles around each center tile (for 3x3 (radius = 1) - macximal is 9  
      min_var =      VARIANCE_THRESHOLD, # Minimal tile variance to include
      max_var =      1000.0,             # Maximal tile variance to include
      min_neibs =    MIN_NEIBS)          # Minimal number of valid tiles to include
      num_gt_test = num_batch_tiles_test.sum()
      high_fract_test = 1.0 * num_gt_test / (num_le_test + num_gt_test)
      print("Number of > %f disparity variance tiles: %d, fraction = %f (test)"%(VARIANCE_THRESHOLD, num_gt_test, high_fract_test))
      fpath =  test_filenameTFR +("TEST_R%d_GT%4.1f"%(RADIUS,VARIANCE_THRESHOLD))
      ex_data.writeTFRewcordsEpoch(fpath, ml_list = ml_list_test, files_list = ex_data.files_test, set_ds= ex_data.test_ds, radius = RADIUS)
  plt.show()
  
  pass
    
