#!/usr/bin/env python3

__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "andrey@elphel.com"


from PIL import Image

import os
import sys
import glob

import numpy as np

import time

import matplotlib.pyplot as plt

import qcstereo_functions as qsf

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
    
globals().update(parameters)


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
figs = []
def setlimsxy(lim_xy):
    if not lim_xy is None:
        plt.xlim(min(lim_xy[:2]),max(lim_xy[:2]))            
        plt.ylim(max(lim_xy[2:]),min(lim_xy[2:]))            
    
for nfile, fpars in enumerate(fig_params):
    if not fpars is None:
        data = qsf.result_npy_prepare(files['result'][nfile], ABSOLUTE_DISPARITY, fix_nan=True, insert_deltas=True)
        
        for rng in fpars['ranges']:
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
            pass

#whow to allow adjustment before applying tight_layout?
pass
for fig in figs:
    fig.tight_layout(rect =[0,0,1,TIGHT_TOP], h_pad = TIGHT_HPAD, w_pad = TIGHT_WPAD)
plt.show()



#qsf.evaluateAllResults(result_files = files['result'],
#                       absolute_disparity = ABSOLUTE_DISPARITY,
#                       cluster_radius = CLUSTER_RADIUS)
print("All done")
exit (0)
