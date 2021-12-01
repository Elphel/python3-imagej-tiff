#!/usr/bin/env python3

__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "andrey@elphel.com"


#from PIL import Image

import os
import sys
#import glob

#import numpy as np

import imagej_tiffwriter

import time

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import qcstereo_functions as qsf
import numpy as np

#import xml.etree.ElementTree as ET

qsf.TIME_START = time.time()
qsf.TIME_LAST  = qsf.TIME_START

IMG_WIDTH =        324 # tiles per image row
DEBUG_LEVEL= 1

try:
    conf_file =  sys.argv[1]
except IndexError:
    print("Configuration path is required as a first argument. Optional second argument specifies root directory for data files")
    exit(1)
try:
    root_dir =  sys.argv[2]
except IndexError:
    root_dir =  os.path.dirname(conf_file)
    
print ("Configuration file: " + conf_file)
parameters, dirs, files, dbg_parameters = qsf.parseXmlConfig(conf_file, root_dir)
"""
Temporarily for backward compatibility
"""
if not "SLOSS_CLIP" in parameters:
    parameters['SLOSS_CLIP'] = 0.5
    print ("Old config, setting SLOSS_CLIP=", parameters['SLOSS_CLIP'])
    
"""
Defined in config file
"""
TILE_SIDE, TILE_LAYERS, TWO_TRAINS, NET_ARCH1, NET_ARCH2 = [None]*5
ABSOLUTE_DISPARITY,SYM8_SUB, WLOSS_LAMBDA,  SLOSS_LAMBDA, SLOSS_CLIP  = [None]*5
SPREAD_CONVERGENCE, INTER_CONVERGENCE, HOR_FLIP, DISP_DIFF_CAP, DISP_DIFF_SLOPE  = [None]*5
CLUSTER_RADIUS,ABSOLUTE_DISPARITY = [None]*2
FIGS_EXTENSIONS = ['png','pdf','svg']
#FIGS_ESXTENSIONS = ['png','pdf','svg']
EVAL_MODES = ["train","infer"]
FIGS_SAVESHOW = ['save','show']

    
globals().update(parameters)
try:
    FIGS_EXTENSIONS = globals()['FIGS_ESXTENSIONS'] # fixing typo in configs
except:
    pass


#exit(0)


TILE_SIZE =         TILE_SIDE* TILE_SIDE # == 81
FEATURES_PER_TILE =  TILE_LAYERS * TILE_SIZE# == 324
BATCH_SIZE =       ([1,2][TWO_TRAINS])*2*1000//25 # == 80 Each batch of tiles has balanced D/S tiles, shuffled batches but not inside batches

SUFFIX=(str(NET_ARCH1)+'-'+str(NET_ARCH2)+
       (["R","A"][ABSOLUTE_DISPARITY]) +
       (["NS","S8"][SYM8_SUB])+
       "WLAM"+str(WLOSS_LAMBDA)+
       "SLAM"+str(SLOSS_LAMBDA)+
       "SCLP"+str(SLOSS_CLIP)+
       (['_nG','_G'][SPREAD_CONVERGENCE])+
       (['_nI','_I'][INTER_CONVERGENCE]) +
       (['_nHF',"_HF"][HOR_FLIP]) +
       ('_CP'+str(DISP_DIFF_CAP)) +
       ('_S'+str(DISP_DIFF_SLOPE))
       )

##############################################################################
cluster_size = (2 * CLUSTER_RADIUS + 1) * (2 * CLUSTER_RADIUS + 1)
center_tile_index = 2 * CLUSTER_RADIUS * (CLUSTER_RADIUS + 1)
qsf.prepareFiles(dirs, files, suffix = SUFFIX)

#import tensorflow.contrib.slim as slim
NN_DISP =   0
HEUR_DISP = 1
GT_DISP =   2
GT_CONF =   3
NN_NAN =    4
HEUR_NAN =  5
NN_DIFF =   6
HEUR_DIFF = 7
CONF_MAX = 0.7
ERR_AMPL = 0.3
TIGHT_TOP = 0.95
TIGHT_HPAD = 1.0
TIGHT_WPAD = 1.0
FIGSIZE = [8.5,11.0]
WOI_COLOR = "red"
TRANSPARENT = True # for export

#dbg_parameters
def get_fig_params(disparity_ranges):
    fig_params = []
    for dr in disparity_ranges:
        if dr[-1][0]=='-':
            fig_params.append(None)
        else:
            subs = []
            for s in dr[:-1]:
                mm = s[:2]
                try:
                    lims = s[2]
                except IndexError:
                    lims = None
                subs.append({'lim_val':mm, 'lim_xy':lims})        
            fig_params.append({'name':dr[-1],'ranges':subs}) 
    return fig_params

#try:
fig_params = get_fig_params(dbg_parameters['disparity_ranges'])

pass

#temporary:
TIFF_ONLY = False # True
#max_bad =        2.5 # excludes only direct bad
max_bad =        2.5 #2.5 # 1.5 # excludes only direct bad
max_diff =       1.5 # 2.0 # 5.0 # maximal max-min difference
max_target_err = 1.0 # 0.5 # maximal max-min difference
max_disp =       5.0

min_strength =   0.18 #ignore tiles below
min_neibs =      1
max_log_to_mm =  0.5 # difference between center average and center should be under this fraction of max-min (0 - disables feature) 


#num_bins = 256 # number of histogram bins
num_bins =        15 # 50 # number of histogram bins
use_gt_weights = True # False # True
index_gt =         2
index_gt_weight =  3
index_heur_err =   7
index_nn_err =     6
index_mm =         8 # max-min
index_log =        9
index_bad =       10
index_num_neibs = 11
"""
Debugging high 9-tile variations, removing error for all tiles with lower difference between max and min
"""
#min_diff =       0.25 # remove all flat tiles with spread less than this (do not show on heuristic/network disparity errors subplots 
min_diff =       0 # remove all flat tiles with spread less than this 




max_target_err2 = max_target_err * max_target_err
if not 'show' in FIGS_SAVESHOW:
    plt.ioff()

#for mode in ['train','infer']:
for mode in ['infer']:
    figs = []
    ffiles = [] # no ext
    def setlimsxy(lim_xy):
        if not lim_xy is None:
            plt.xlim(min(lim_xy[:2]),max(lim_xy[:2]))            
            plt.ylim(max(lim_xy[2:]),min(lim_xy[2:]))
    cumul_weights = None                   
        
    for nfile, fpars in enumerate(fig_params):
        if not fpars is None:
            img_file = files['result'][nfile]
            if mode == 'infer':
                img_file = img_file.replace('.npy','-infer.npy')
            """    
            try:    
#                data,_ = qsf.result_npy_prepare(img_file, ABSOLUTE_DISPARITY, fix_nan=True, insert_deltas=True)
#                data,_ = qsf.result_npy_prepare(img_file, ABSOLUTE_DISPARITY, fix_nan=True, insert_deltas=3)
                data,labels = qsf.result_npy_prepare(img_file, ABSOLUTE_DISPARITY, fix_nan=True, insert_deltas=3)
            except:
                print ("Image file does not exist:", img_file)
                continue
            """
            pass
            data,labels = qsf.result_npy_prepare(img_file, ABSOLUTE_DISPARITY, fix_nan=True, insert_deltas=3)
            if  True: #TIFF_ONLY:
                
                
                tiff_path = img_file.replace('.npy','-test.tiff')
                        
                data = data.transpose(2,0,1)
                print("Saving results to TIFF: "+tiff_path)
                imagej_tiffwriter.save(tiff_path,data,labels=labels)
                """
                Calculate histograms
                """
                err_heur2 = data[index_heur_err]*data[index_heur_err] 
                err_nn2 =   data[index_nn_err]*  data[index_nn_err] 
                diff_log2 = data[index_log]*     data[index_log] 
                weights = (
                    (data[index_gt] < max_disp) & 
                    (err_heur2 < max_target_err2) &
                    (data[index_bad] < max_bad) &
                    (data[index_gt_weight] >= min_strength) &
                    (data[index_num_neibs] >= min_neibs)&
#max_log_to_mm =  0.5 # difference between center average and center should be under this fraction of max-min (0 - disables feature) 
                    (data[index_log] < max_log_to_mm * np.sqrt(data[index_mm]) )                    
                    ).astype(data.dtype) # 0.0/1.1
                #max_disp
                
                #max_target_err
                if  use_gt_weights:
                    weights *= data[index_gt_weight]
                mm =     data[index_mm]
                weh = np.nan_to_num(weights*err_heur2)
                wen = np.nan_to_num(weights*err_nn2)
                wel = np.nan_to_num(weights*diff_log2)
                hist_weights,bin_vals =   np.histogram(a=mm, bins = num_bins, range = (0.0, max_diff), weights = weights,  density = False)
                hist_err_heur2,_ = np.histogram(a=mm, bins = num_bins, range = (0.0, max_diff), weights = weh,      density = False)
                hist_err_nn2,_ =   np.histogram(a=mm, bins = num_bins, range = (0.0, max_diff), weights = wen,      density = False)
                hist_diff_log2,_ = np.histogram(a=mm, bins = num_bins, range = (0.0, max_diff), weights = wel,      density = False)
                if cumul_weights is None:
                    cumul_weights =    hist_weights
                    cumul_err_heur2 =  hist_err_heur2
                    cumul_err_nn2 =    hist_err_nn2
                    cumul_diff_log2 =  hist_diff_log2
                else:
                    cumul_weights +=   hist_weights
                    cumul_err_heur2 += hist_err_heur2
                    cumul_err_nn2 +=   hist_err_nn2
                    cumul_diff_log2 += hist_diff_log2
                
                hist_err_heur2 =   np.nan_to_num(hist_err_heur2/hist_weights)
                hist_err_nn2 =     np.nan_to_num(hist_err_nn2/hist_weights)
                hist_gain2 = np.nan_to_num(hist_err_heur2/hist_err_nn2)
                hist_gain = np.sqrt(hist_gain2)
                hist_diff_log2 =   np.nan_to_num(hist_diff_log2/hist_weights)

                print("hist_err_heur2", end = " ")
                print(np.sqrt(hist_err_heur2))
                print("hist_err_nn2", end = " ")
                print(np.sqrt(hist_err_nn2))
                print("hist_gain", end = " ")
                print(hist_gain)
                print("hist_diff_log2", end = " ")
                print(np.sqrt(hist_diff_log2))
                
                
                if min_diff> 0.0:
                    pass
                    good = (mm > min_diff).astype(mm.dtype)
                    good /= good # good - 1, bad - nan
                    data[index_heur_err] *= good
                    data[index_nn_err] *= good
                data = data.transpose(1,2,0)
                
            if TIFF_ONLY:
                continue
        
        
            for subindex, rng in enumerate(fpars['ranges']):
                lim_val = rng['lim_val']
                lim_xy =  rng['lim_xy']
                fig = plt.figure(figsize=FIGSIZE)
                fig.canvas.set_window_title(fpars['name'])
                fig.suptitle(fpars['name'])
                ax_conf=plt.subplot(322)
                ax_conf.set_title("Ground truth confidence")
    #            fig.suptitle("Groud truth confidence")
                plt.imshow(data[...,GT_CONF], vmin=0, vmax=CONF_MAX, cmap='gray')
                if not lim_xy is None:
                    pass # show frame
                    xdata=[min(lim_xy[:2]),max(lim_xy[:2]),max(lim_xy[:2]),min(lim_xy[:2]),min(lim_xy[:2])]
                    ydata=[min(lim_xy[2:]),min(lim_xy[2:]),max(lim_xy[2:]),max(lim_xy[2:]),min(lim_xy[2:])]
                    plt.plot(xdata,ydata,color=WOI_COLOR)
                
    #            setlimsxy(lim_xy)
                plt.colorbar(orientation='vertical') # location='bottom')
                
                ax_gtd=plt.subplot(321)
                ax_gtd.set_title("Ground truth disparity map")
                plt.imshow(data[...,GT_DISP], vmin=lim_val[0], vmax=lim_val[1])
                setlimsxy(lim_xy)            
                plt.colorbar(orientation='vertical') # location='bottom')
                
                ax_hed=plt.subplot(323)
                ax_hed.set_title("Heuristic disparity map")
                plt.imshow(data[...,HEUR_NAN], vmin=lim_val[0], vmax=lim_val[1])            
                setlimsxy(lim_xy)                        
                plt.colorbar(orientation='vertical') # location='bottom')
            
                ax_nnd=plt.subplot(325)
                ax_nnd.set_title("Network disparity output")
                plt.imshow(data[...,NN_NAN], vmin=lim_val[0], vmax=lim_val[1])
                setlimsxy(lim_xy)                        
                plt.colorbar(orientation='vertical') # location='bottom')
    
                ax_hee=plt.subplot(324)
                ax_hee.set_title("Heuristic disparity error")
                plt.imshow(data[...,HEUR_DIFF], vmin=-ERR_AMPL, vmax=ERR_AMPL)            
                setlimsxy(lim_xy)                        
                plt.colorbar(orientation='vertical') # location='bottom')
    
                ax_nne=plt.subplot(326)
                ax_nne.set_title("Network disparity error")
                plt.imshow(data[...,NN_DIFF], vmin=-ERR_AMPL, vmax=ERR_AMPL)            
                setlimsxy(lim_xy)                        
                plt.colorbar(orientation='vertical') # location='bottom')
            
                plt.tight_layout(rect =[0,0,1,TIGHT_TOP], h_pad = TIGHT_HPAD, w_pad = TIGHT_WPAD)
                
                figs.append(fig)
                fb_noext = os.path.splitext(os.path.basename(img_file))[0]#
                if subindex > 0:
                    if subindex < 10:
                        fb_noext+="abcdefghi"[subindex-1]
                    else:
                        fb_noext+="-"+str(subindex)
                ffiles.append(fb_noext)
                pass
    if True:
        cumul_err_heur2 =   np.nan_to_num(cumul_err_heur2/cumul_weights)
        cumul_err_nn2 =     np.nan_to_num(cumul_err_nn2/cumul_weights)
        cumul_gain2 =       np.nan_to_num(cumul_err_heur2/cumul_err_nn2)
        cumul_gain =        np.sqrt(cumul_gain2)
        cumul_diff_log2 =   np.nan_to_num(cumul_diff_log2/cumul_weights)
        print("cumul_weights", end = " ")
        print(cumul_weights)
        print("cumul_err_heur", end = " ")
        print(np.sqrt(cumul_err_heur2))
        print("cumul_err_nn", end = " ")
        print(np.sqrt(cumul_err_nn2))
        print("cumul_gain", end = " ")
        print(cumul_gain)
        print("cumul_diff_log2", end = " ")
        print(np.sqrt(cumul_diff_log2))
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('3x3 tiles ground truth disparity max-min (pix)')
        ax1.set_ylabel('RMSE\n(pix)', color='black', rotation='horizontal')
        ax1.yaxis.set_label_coords(-0.045,0.92)
        
        ax1.plot(bin_vals[0:-1], np.sqrt(cumul_err_nn2),   'tab:red',label="network disparity RMSE")
        ax1.plot(bin_vals[0:-1], np.sqrt(cumul_err_heur2), 'tab:green',label="heuristic disparity RMSE")
        ax1.plot(bin_vals[0:-1], np.sqrt(cumul_diff_log2), 'tab:cyan',label="ground truth LoG")
        
        ax1.tick_params(axis='y', labelcolor='black')
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('weight', color='black', rotation='horizontal')  # we already handled the x-label with ax1  
        ax2.yaxis.set_label_coords(1.06,1.0)
        
        
              
        ax2.plot(bin_vals[0:-1], cumul_weights,color='grey',dashes=[6, 2],label='weights = n_tiles * gt_confidence')
        ax1.legend(loc="upper left", bbox_to_anchor=(0.2,1.0))
        ax2.legend(loc="lower right", bbox_to_anchor=(1.0,0.1))
        
        """
    
        fig = plt.figure(figsize=FIGSIZE)
        fig.canvas.set_window_title('Cumulative')
        fig.suptitle('Difference to GT')
#        ax_conf=plt.subplot(322)
        ax_conf=plt.subplot(211)
        ax_conf.set_title("RMS vs max9-min9")
        plt.plot(bin_vals[0:-1], np.sqrt(cumul_err_heur2),'red',
                 bin_vals[0:-1], np.sqrt(cumul_err_nn2),'green',
                 bin_vals[0:-1], np.sqrt(cumul_diff_log2),'blue')
        figs.append(fig)
        ffiles.append('cumulative')
        ax_conf=plt.subplot(212)
        ax_conf.set_title("weights vs max9-min9")
        plt.plot(bin_vals[0:-1], cumul_weights,'black')
        """
        figs.append(fig)
        ffiles.append('cumulative')
        pass
        #bin_vals[0:-1]
        
#            fig.suptitle("Groud truth confidence")

#    
    #whow to allow adjustment before applying tight_layout?
    pass
    for fig in figs:
        fig.tight_layout(rect =[0,0,1,TIGHT_TOP], h_pad = TIGHT_HPAD, w_pad = TIGHT_WPAD)
    
    if FIGS_EXTENSIONS and figs and 'save' in FIGS_SAVESHOW:
        try:
            print ("Creating output directory for figures: ",dirs['figures'])
            os.makedirs(dirs['figures'])
        except:
            pass
        pp=None
        if 'pdf' in FIGS_EXTENSIONS:
            if mode == 'infer':
                pdf_path = os.path.join(dirs['figures'],"figures-infer%s.pdf"%str(min_diff))
            else:
                pdf_path = os.path.join(dirs['figures'],"figures-train%s.pdf"%str(min_diff))
            pp= PdfPages(pdf_path)
        
        for fb_noext, fig in zip(ffiles,figs):
            for ext in FIGS_EXTENSIONS:
                if ext == 'pdf':
                    pass
                    fig.savefig(pp,format='pdf')
                else:
                    if mode == 'infer':
                        noext = fb_noext+'-infer'
                    else:
                        noext = fb_noext+'-train'
                    
                    fig.savefig(
                        fname = os.path.join(dirs['figures'],noext+"."+ext),
                        transparent = TRANSPARENT,
                        )
                pass
        if pp:
            pp.close()
            
            
if 'show' in FIGS_SAVESHOW:    
    plt.show()

#FIGS_ESXTENSIONS

qsf.evaluateAllResults(result_files = files['result'],
                       absolute_disparity = ABSOLUTE_DISPARITY,
                       cluster_radius = CLUSTER_RADIUS)
print("All done")
exit (0)
