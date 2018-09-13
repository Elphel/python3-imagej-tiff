#!/usr/bin/env python3
__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "andrey@elphel.com"

import os
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import time
import imagej_tiffwriter
TIME_LAST = 0
TIME_START = 0

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
def print_time(txt="",end="\n"):
    global TIME_LAST
    t = time.time()
    if txt:
        txt +=" "
    print(("%s"+bcolors.BOLDWHITE+"at %.4fs (+%.4fs)"+bcolors.ENDC)%(txt,t-TIME_START,t-TIME_LAST), end = end, flush=True)
    TIME_LAST = t

DEFAULT_TITLES = [
        ['test_lvar','Test_flat_heuristic'],
        ['test_hvar','Test_edge_heuristic'],
        ['test_lvar1','Test_flat_random'],
        ['test_hvar1','Test_edge_random'],
        ['fake_lvar','Fake_flat_heuristic'],
        ['fake_hvar','Fake_edge_heuristic'],
        ['fake_lvar1','Fake_flat_random'],
        ['fake_hvar1','Fake_edge_random']]
    
def parseXmlConfig(conf_file, root_dir):
    tree = ET.parse(conf_file)
    root = tree.getroot()
    parameters = {}
    for p in root.find('parameters'):
##        print ("p.tag=%s,  p.text.stri[p()=%s"%(p.tag,p.text.strip()))
        parameters[p.tag]=eval(p.text.strip())
#    globals    
    dirs={}
    for p in root.find('directories'):
        dirs[p.tag]=eval(p.text.strip())
        if not os.path.isabs(dirs[p.tag]):
            dirs[p.tag] = os.path.join(root_dir, dirs[p.tag])
    files={}
    for p in root.find('files'):
        files[p.tag]=eval(p.text.strip())
    dbg_parameters = {}
    for p in root.find('dbg_parameters'):
        dbg_parameters[p.tag]=eval(p.text.strip())

    return parameters, dirs, files, dbg_parameters

def defaultTestTitles(files):
    test_titles = []
    for f, n in DEFAULT_TITLES:
        if f in files:
            test_titles.append(n)
        else:
            test_titles.append(None)
    return test_titles


def prepareFiles(dirs, files, suffix):
    #MAX_FILES_PER_GROUP
    for i, path in enumerate(files['train_lvar']):
        files['train_lvar'][i]=os.path.join(dirs['train_lvar'], path)
        
    for i, path in enumerate(files['train_hvar']):
        files['train_hvar'][i]=os.path.join(dirs['train_hvar'], path)
    
    for i, path in enumerate(files['train_lvar1']):
        files['train_lvar1'][i]=os.path.join(dirs['train_lvar1'], path)
        
    for i, path in enumerate(files['train_hvar1']):
        files['train_hvar1'][i]=os.path.join(dirs['train_hvar1'], path)
        
    for i, path in enumerate(files['test_lvar']):
        files['test_lvar'][i]=os.path.join(dirs['test_lvar'], path)
        
    for i, path in enumerate(files['test_hvar']):
        files['test_hvar'][i]=os.path.join(dirs['test_hvar'], path)
    if ('test_lvar1' in files) and ('test_lvar1' in dirs):
        for i, path in enumerate(files['test_lvar1']):
            files['test_lvar1'][i]=os.path.join(dirs['test_lvar1'], path)
    if ('test_hvar1' in files) and ('test_hvar1' in dirs):
        for i, path in enumerate(files['test_hvar1']):
            files['test_hvar1'][i]=os.path.join(dirs['test_hvar1'], path)

    if ('fake_lvar' in files) and ('fake_lvar' in dirs):
        for i, path in enumerate(files['fake_lvar']):
            files['fake_lvar'][i]=os.path.join(dirs['fake_lvar'], path)
    if ('fake_hvar' in files) and ('fake_hvar' in dirs):
        for i, path in enumerate(files['fake_hvar']):
            files['fake_hvar'][i]=os.path.join(dirs['fake_hvar'], path)

    if ('fake_lvar1' in files) and ('fake_lvar1' in dirs):
        for i, path in enumerate(files['fake_lvar1']):
            files['fake_lvar1'][i]=os.path.join(dirs['fake_lvar1'], path)
    if ('fake_hvar' in files) and ('fake_hvar' in dirs):
        for i, path in enumerate(files['fake_hvar1']):
            files['fake_hvar1'][i]=os.path.join(dirs['fake_hvar1'], path)

         
        
    result_files=[]
    for i, path in enumerate(files['images']):
        result_files.append(os.path.join(dirs['result'], path+"_"+suffix+'.npy'))
    files['result'] = result_files

    if not 'checkpoints' in files:
        files['checkpoints'] = 'checkpoints'
    if not 'checkpoints' in dirs:
        dirs['checkpoints'] = dirs['result']
    files['checkpoints'] = os.path.join(dirs['checkpoints'], files['checkpoints'])    

    if not 'inference' in files:
        files['inference'] = 'inference'
    if not 'inference' in dirs:
        dirs['inference'] = dirs['result']
    files['inference'] = os.path.join(dirs['inference'], files['inference'])    
    
    if not 'figures' in dirs:
        dirs['figures'] = os.path.join(dirs['result'],"figs")
     
    files['train'] = [files['train_lvar'],files['train_hvar'], files['train_lvar1'], files['train_hvar1']]
    files['test'] =  [files['test_lvar'], files['test_hvar']]
    if 'test_lvar1' in files:
        files['test'].append(files['test_lvar1'])
    if 'test_hvar1' in files:
        files['test'].append(files['test_hvar1'])
    # should be after result files
    for i, path in enumerate(files['images']):
        files['images'][i] =      os.path.join(dirs['images'],    path+'.tfrecords')

def readTFRewcordsEpoch(train_filename, cluster_radius):
    if not  '.tfrecords' in train_filename:
        train_filename += '.tfrecords'
    npy_dir_name = "npy"
    dirname = os.path.dirname(train_filename) 
    npy_dir = os.path.join(dirname, npy_dir_name)
    filebasename, _ = os.path.splitext(train_filename)
    filebasename = os.path.basename(filebasename)
    file_all =     os.path.join(npy_dir,filebasename + '.npy')
    if  os.path.exists(file_all):
        data =             np.load (file_all)
    else:     
        record_iterator = tf.python_io.tf_record_iterator(path=train_filename)
        corr2d_list=[]
        target_disparity_list=[]
        gt_ds_list = []
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            corr2d_list.append           (np.array(example.features.feature['corr2d'].float_list.value, dtype=np.float32))
            target_disparity_list.append (np.array(example.features.feature['target_disparity'].float_list.value, dtype=np.float32))
            gt_ds_list.append            (np.array(example.features.feature['gt_ds'].float_list.value, dtype= np.float32))
            pass
        corr2d=            np.array(corr2d_list)
        target_disparity = np.array(target_disparity_list)
        gt_ds =            np.array(gt_ds_list)
        try:
            os.makedirs(os.path.dirname(file_all))
        except:
            pass
        if cluster_radius > 0:
            reformat_to_clusters(corr2d, target_disparity, gt_ds, cluster_radius)
        data = np.concatenate([corr2d, target_disparity, gt_ds],axis = 1)
        np.save(file_all, data)
    
    return data


def getMoreFiles(fpaths,rslt, cluster_radius, hor_flip, tile_layers, tile_side):
    for fpath in fpaths:
        dataset = readTFRewcordsEpoch(fpath, cluster_radius)
        
        if hor_flip:
            if np.random.randint(2):
                print_time("Performing horizontal flip", end=" ")
                flip_horizontal(dataset, cluster_radius, tile_layers, tile_side)
                print_time("Done")
        rslt.append(dataset)

#from http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
def read_and_decode(filename_queue, featrures_per_tile):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'corr2d':           tf.FixedLenFeature([featrures_per_tile],tf.float32), #string),
        'target_disparity': tf.FixedLenFeature([1],   tf.float32), #.string),
        'gt_ds':            tf.FixedLenFeature([2],  tf.float32)  #.string)
        })
    corr2d =           features['corr2d'] # tf.decode_raw(features['corr2d'], tf.float32)
    target_disparity = features['target_disparity'] # tf.decode_raw(features['target_disparity'], tf.float32)
    gt_ds =            tf.cast(features['gt_ds'], tf.float32) # tf.decode_raw(features['gt_ds'], tf.float32)
    in_features = tf.concat([corr2d,target_disparity],0)
    corr2d_out, target_disparity_out, gt_ds_out = tf.train.shuffle_batch( [in_features, target_disparity, gt_ds],
                                                 batch_size=1000, # 2,
                                                 capacity=30,
                                                 num_threads=2,
                                                 min_after_dequeue=10)
    return corr2d_out, target_disparity_out, gt_ds_out
def add_margins(npa,radius, val = np.nan):
    npa_ext = np.empty((npa.shape[0]+2*radius, npa.shape[1]+2*radius, npa.shape[2]), dtype = npa.dtype)
    npa_ext[radius:radius + npa.shape[0],radius:radius + npa.shape[1]] = npa
    npa_ext[0:radius,:,:] = val  
    npa_ext[radius + npa.shape[0]:,:,:] = val  
    npa_ext[:,0:radius,:] = val  
    npa_ext[:, radius + npa.shape[1]:,:] = val
    return npa_ext   

def add_neibs(npa_ext,radius):
    height = npa_ext.shape[0]-2*radius
    width =  npa_ext.shape[1]-2*radius
    side = 2 * radius + 1
#    size = side * side
    npa_neib = np.empty((height,  width, side, side, npa_ext.shape[2]), dtype = npa_ext.dtype)
    for dy in range (side):
        for dx in range (side):
            npa_neib[:,:,dy, dx,:]= npa_ext[dy:dy+height, dx:dx+width]
    return npa_neib.reshape(height, width, -1)    
  
def extend_img_to_clusters(datasets_img,radius, width): #  = 324):
#    side = 2 * radius + 1
#    size = side * side
    if len(datasets_img) ==0:
        return
    num_tiles = datasets_img[0]['corr2d'].shape[0]
    height = num_tiles // width 
    for rec in datasets_img:
        if not rec is None:
            rec['corr2d'] =           add_neibs(add_margins(rec['corr2d'].reshape((height,width,-1)), radius, np.nan), radius).reshape((num_tiles,-1)) 
            rec['target_disparity'] = add_neibs(add_margins(rec['target_disparity'].reshape((height,width,-1)), radius, np.nan), radius).reshape((num_tiles,-1)) 
            rec['gt_ds'] =            add_neibs(add_margins(rec['gt_ds'].reshape((height,width,-1)), radius, np.nan), radius).reshape((num_tiles,-1))
            pass
def reformat_to_clusters_rec(datasets_data, cluster_radius):
    cluster_size = (2 * cluster_radius + 1) * (2 * cluster_radius + 1)
# Reformat input data
    for rec in datasets_data:
        rec['corr2d'] =           rec['corr2d'].reshape(          (rec['corr2d'].shape[0]//cluster_size,           rec['corr2d'].shape[1] * cluster_size)) 
        rec['target_disparity'] = rec['target_disparity'].reshape((rec['target_disparity'].shape[0]//cluster_size, rec['target_disparity'].shape[1] * cluster_size)) 
        rec['gt_ds'] =            rec['gt_ds'].reshape(           (rec['gt_ds'].shape[0]//cluster_size,            rec['gt_ds'].shape[1] * cluster_size))
def reformat_to_clusters(corr2d, target_disparity, gt_ds, cluster_radius):
    cluster_size = (2 * cluster_radius + 1) * (2 * cluster_radius + 1)
# Reformat input data
    corr2d.shape =           ((corr2d.shape[0]//cluster_size,           corr2d.shape[1] * cluster_size)) 
    target_disparity.shape = ((target_disparity.shape[0]//cluster_size, target_disparity.shape[1] * cluster_size)) 
    gt_ds.shape =            ((gt_ds.shape[0]//cluster_size,            gt_ds.shape[1] * cluster_size))

def get_lengths(cluster_radius, tile_layers, tile_side):
    cluster_side = 2 * cluster_radius + 1
    cl = cluster_side * cluster_side * tile_layers * tile_side * tile_side
    tl = cluster_side * cluster_side
    return  cl, tl,cluster_side


def flip_horizontal(dataset, cluster_radius, tile_layers, tile_side):
    cl,tl,cluster_side = get_lengths(cluster_radius, tile_layers, tile_side) 
    corr2d =           dataset[:,:cl]     .reshape([dataset.shape[0],  cluster_side, cluster_side, tile_layers, tile_side, tile_side])
    target_disparity = dataset[:,cl:cl+tl].reshape([dataset.shape[0],  cluster_side, cluster_side, -1])
    gt_ds =            dataset[:,cl+tl:]  .reshape([dataset.shape[0],  cluster_side, cluster_side, -1])

    """
    Horizontal flip of tiles
    """
    corr2d = corr2d[:,:,::-1,...]
    target_disparity = target_disparity[:,:,::-1,...]
    gt_ds = gt_ds[:,:,::-1,...]
    
    corr2d[:,:,:,0,:,:] = corr2d[:,:,:,0,::-1,:] # flip vertical layer0   (hor) 
    corr2d[:,:,:,1,:,:] = corr2d[:,:,:,1,:,::-1]  # flip horizontal layer1 (vert)
    corr2d_2 =            corr2d[:,:,:,3,::-1,:].copy() # flip vertical layer3   (diago)
    corr2d[:,:,:,3,:,:] = corr2d[:,:,:,2,::-1,:] # flip vertical layer2   (diago)
    corr2d[:,:,:,2,:,:] = corr2d_2
    """
    pack back into a single (input)array
    """
    dataset[:,:cl] =      corr2d.reshape((corr2d.shape[0],-1))
    dataset[:,cl:cl+tl] = target_disparity.reshape((target_disparity.shape[0],-1)) 
    dataset[:,cl+tl:] =   gt_ds.reshape((gt_ds.shape[0],-1))

def replace_nan(datasets_data): # , cluster_radius):
# Reformat input data
    for rec in datasets_data:
        if not rec is None:
            np.nan_to_num(rec['corr2d'],           copy = False) 
            np.nan_to_num(rec['target_disparity'], copy = False)
            if  'gt_ds' in rec:
                np.nan_to_num(rec['gt_ds'],        copy = False)

def permute_to_swaps(perm):
    pairs = []
    for i in range(len(perm)):
        w = np.where(perm == i)[0][0]
        if w != i:
            pairs.append([i,w])
            perm[w] = perm[i]
            perm[i] = i
    return pairs        
        

def shuffle_in_place(dataset_data, #alternating clusters from 4 sources.each cluster has all needed data (concatenated)
                      period):
    for i in range (period):
        np.random.shuffle(dataset_data[i::period]) 

def add_file_to_dataset(dataset, new_dataset, train_next):
    train_next['file'] = (train_next['file']+1)%train_next['files'] 
    l = new_dataset.shape[0] * train_next['step']
    if (train_next['entry'] + l) < (train_next['entries']+train_next['step']):
        dataset[train_next['entry']:train_next['entry']+l:train_next['step']] = new_dataset
        train_next['entry'] += l
        if (train_next['entry'] >= train_next['entries']):
            train_next['entry'] -= train_next['entries']
            return True
        else:
            return False
    else: # split it two parts
        l = (train_next['entries'] - train_next['entry'] + (train_next['step']-1)) // train_next['step']
        dataset[train_next['entry']::train_next['step']] = new_dataset[:l]
        
        train_next['entry'] = (train_next['entry'] + l * train_next['step']) % train_next['entries'] #0,1,2,3
        
        l1 = new_dataset.shape[0] - l # remainder
        ln = train_next['entry'] + l1 * train_next['step']
        dataset[train_next['entry']:ln:train_next['step']] = new_dataset[l:]
        train_next['entry'] = ln 
        return True

"""
train_next[n_train]
Read as many files as needed, possibly repeating, until each buffer is f
"""    

def initTrainTestData(
        files,
        cluster_radius,
        buffer_size, # number of clusters per train
        test_titles = None
        ):
    """
    Generates a single np array for training with concatenated cluster of corr2d,
    cluster of target_disparity, and cluster of gt_ds for convenient shuffling                  
    
    """
    num_trains = len(files['train'])
    num_entries = num_trains * buffer_size  
    dataset_train_merged = None
    train_next = [None]*num_trains
    for n_train, f_train in enumerate(files['train']):
        train_next[n_train] = {'file':0, 'entry':n_train, 'files':len(f_train), 'entries': num_entries, 'step':num_trains, 'more_files':False}
        buffer_full = False
        while not buffer_full:
            for fpath in f_train:
                print_time("Importing train data "+(["low variance","high variance", "low variance1","high variance1"][n_train]) +" from "+fpath, end="")
                new_dataset = readTFRewcordsEpoch(fpath, cluster_radius)
                if dataset_train_merged is None:
                    dataset_train_merged = np.empty([num_entries,new_dataset.shape[1]], dtype =new_dataset.dtype)
##                print("\nbefore add_file_to_dataset: train_next["+str(n_train)+"]=",train_next[n_train])
                rollover = add_file_to_dataset(
                    dataset = dataset_train_merged,
                    new_dataset = new_dataset,
                    train_next = train_next[n_train])
##                print("after add_file_to_dataset: train_next["+str(n_train)+"]=",train_next[n_train])
                print_time("  Done")
                if rollover:
                    buffer_full = True
                    train_next[n_train][ 'more_files'] = train_next[n_train][ 'file'] < train_next[n_train][ 'files'] # Not all files used, need to load during training
                    break
    if test_titles is None:
        test_titles = defaultTestTitles(files)
    datasets_test = []
    for t,v in zip(test_titles,DEFAULT_TITLES):
        if not t is None:
            grp = v[0]
            for fpath in files[grp]:
                print_time("Importing test data ("+grp+") from "+fpath, end="")
                new_dataset = readTFRewcordsEpoch(fpath, cluster_radius)
                datasets_test.append(new_dataset)
                print_time("  Done")
    """
    for grp in ['test_lvar','test_hvar','test_lvar1','test_hvar1']:
        if grp in files:
            for fpath in files[grp]:
                print_time("Importing test data ("+grp+") from "+fpath, end="")
                new_dataset = readTFRewcordsEpoch(fpath, cluster_radius)
                datasets_test.append(new_dataset)
                print_time("  Done")
    """
    return train_next, dataset_train_merged, datasets_test
        
def get_full_tile_indices2d(height,width):
    a = np.empty([height,width,2], dtype=np.int32)
    a[...,0] = np.arange(height).reshape([height,1])
    a[...,1] = np.arange(width)
    return a.reshape(-1,2)
def get_full_tile_indices(height,width):
    return np.arange(height*width).reshape(-1,1)

def readImageData(image_data,
                  files,
                  indx,
                  cluster_radius,
                  tile_layers,
                  tile_side,
                  width,
                  replace_nans,
                  infer =          False,
                  keep_gt =        False):
    cl,tl,_ = get_lengths(0, tile_layers, tile_side) 
    if image_data[indx] is None:
        dataset = readTFRewcordsEpoch(files['images'][indx], cluster_radius = 0)
        corr2d =           dataset[:,:cl]
        target_disparity = dataset[:,cl:cl+tl]
        if infer:
            image_data[indx] = {
                'corr2d':           corr2d,
                'target_disparity': target_disparity,
                'xy':get_full_tile_indices2d(corr2d.shape[0]//width, width),
                'ntile':get_full_tile_indices(corr2d.shape[0]//width, width)}
            if keep_gt:
                gt_ds =                      dataset[:,cl+tl:]
                image_data[indx]["gt_ds"] =  gt_ds
                image_data[indx]["gtruths"]= gt_ds.copy()
                image_data[indx]["t_disps"]= target_disparity.reshape([-1,1]).copy()
        else:
            gt_ds =            dataset[:,cl+tl:]
            image_data[indx] = {
                'corr2d':           corr2d,
                'target_disparity': target_disparity,
                 "gt_ds":            gt_ds,
                 "gtruths":          gt_ds.copy(),
                 "t_disps":          target_disparity.reshape([-1,1]).copy()}
            if cluster_radius > 0:
                extend_img_to_clusters(
                     [image_data[indx]],
                     cluster_radius,
                     width)
        if replace_nans:
            replace_nan([image_data[indx]])
            
    return image_data[indx]

def initImageData(files,
                  max_imgs,
                  cluster_radius,
                  tile_layers,
                  tile_side,
                  width,
                  replace_nans,
                  infer =          False,
                  keep_gt =        False):
#                  no_train = False):
    num_imgs = len(files['images'])
    img_data = [None] * num_imgs
    for nfile in range(min(num_imgs, max_imgs)):
        print_time("Importing test image data from "+ files['images'][nfile], end="")
        readImageData(img_data,
                      files,
                      nfile,
                      cluster_radius,
                      tile_layers,
                      tile_side,
                      width,
                      replace_nans,
                      infer =     infer,
                      keep_gt =   keep_gt)
        print_time("  Done")
        return img_data
            
def evaluateAllResults(result_files, absolute_disparity, cluster_radius, labels=None, logpath=None):
    if logpath:
        lf=open(logpath,"w")
    else:
        lf = None
    for result_file in result_files:
        try:
            print_time("Reading resuts from "+result_file, end=" ")
            eval_results(result_file, absolute_disparity, radius=cluster_radius, logfile=lf)
        except:
            print_time(" - does not exist")
            continue
        print_time("Done")
        print_time("Saving resuts to tiff", end=" ")
        result_npy_to_tiff(result_file, absolute_disparity, fix_nan = True, labels=labels)        
        print_time("Done")
    if lf:
        lf.close()


def result_npy_prepare(npy_path, absolute, fix_nan, insert_deltas=True,labels=None):
    
    """
    @param npy_path full path to the npy file with 4-layer data (242,324,4) - nn_disparity(offset), target_disparity, gt disparity, gt strength
           data will be written as 4-layer tiff, extension '.npy' replaced with '.tiff'
    @param absolute - True - the first layer contains absolute disparity, False - difference from target_disparity
    @param fix_nan - replace nan in target_disparity with 0 to apply offset, target_disparity will still contain nan
    """
    data = np.load(npy_path) #(324,242,4) [nn_disp, target_disp,gt_disp, gt_conf]
    if labels is None:
        labels = ["chn%d"%(i) for i in range(data.shape[2])]
#    labels = ["nn_out","hier_out","gt_disparity","gt_strength"]
    nn_out =            0
#    target_disparity =  1     
    gt_disparity =      2     
    gt_strength =       3     
    if not absolute:
        if fix_nan:
            data[...,nn_out] +=  np.nan_to_num(data[...,1], copy=True)
        else:
            data[...,nn_out] +=  data[...,1]
    if insert_deltas:
        np.nan_to_num(data[...,gt_strength], copy=False)
        data = np.concatenate([data[...,0:4],data[...,0:2],data[...,0:2],data[...,4:]], axis = 2) # data[...,4:] may be empty
        labels = labels[:4]+["nn_out","hier_out","nn_err","hier_err"]+labels[4:]
        data[...,6] -= data[...,gt_disparity] 
        data[...,7] -= data[...,gt_disparity]
        for l in [2, 4, 5, 6, 7]:
            if l < data.shape[2]:
                data[...,l] = np.select([data[...,gt_strength]==0.0, data[...,gt_strength]>0.0], [np.nan,data[...,l]])
        # All other layers - mast too
        for l in range(8,data.shape[2]):
            data[...,l] = np.select([data[...,gt_strength]==0.0, data[...,gt_strength]>0.0], [np.nan,data[...,l]])
    return data, labels        

def result_npy_to_tiff(npy_path,
                        absolute,
                        fix_nan,
                        insert_deltas=True,
                        labels =      None,
                        logfile =     None):
    
    """
    @param npy_path full path to the npy file with 4-layer data (242,324,4) - nn_disparity(offset), target_disparity, gt disparity, gt strength
           data will be written as 4-layer tiff, extension '.npy' replaced with '.tiff'
    @param absolute - True - the first layer contains absolute disparity, False - difference from target_disparity
    @param fix_nan - replace nan in target_disparity with 0 to apply offset, target_disparity will still contain nan
    """
    data,labels = result_npy_prepare(npy_path, absolute, fix_nan, insert_deltas, labels=labels)
    tiff_path = npy_path.replace('.npy','.tiff')
            
    data = data.transpose(2,0,1)
    print("Saving results to TIFF: "+tiff_path)
    if (logfile):
        print("Saving results to TIFF: "+tiff_path,file=logfile)
        
    imagej_tiffwriter.save(tiff_path,data,labels=labels)        

def eval_results(rslt_path, absolute,
                 min_disp =       -0.1, #minimal GT disparity
                 max_disp =       20.0, # maximal GT disparity
                 max_ofst_target = 1.0,
                 max_ofst_result = 1.0,
                 str_pow =         2.0,
                 radius =          0,
                 logfile = None):
    variants = [[         -0.1,         5.0,              0.5,            0.5,          1.0],           
                [         -0.1,         5.0,              0.5,            0.5,          2.0],
                [         -0.1,         5.0,              0.2,            0.2,          1.0],
                [         -0.1,         5.0,              0.2,            0.2,          2.0],
                [         -0.1,        20.0,              0.5,            0.5,          1.0],           
                [         -0.1,        20.0,              0.5,            0.5,          2.0],
                [         -0.1,        20.0,              0.2,            0.2,          1.0],
                [         -0.1,        20.0,              0.2,            0.2,          2.0],
                [         -0.1,        20.0,              1.0,            1.0,          1.0],
                [min_disp, max_disp, max_ofst_target, max_ofst_result, str_pow]]
    

    rslt = np.load(rslt_path)
    not_nan  =  ~np.isnan(rslt[...,0])
    not_nan &=  ~np.isnan(rslt[...,1])
    not_nan &=  ~np.isnan(rslt[...,2])
    not_nan &=  ~np.isnan(rslt[...,3])
    not_nan_ext = np.zeros((rslt.shape[0] + 2*radius,rslt.shape[1] + 2 * radius),dtype=np.bool) 
    not_nan_ext[radius:-radius,radius:-radius] = not_nan
    for dy in range(2*radius+1):
        for dx in range(2*radius+1):
            not_nan_ext[dy:dy+not_nan.shape[0], dx:dx+not_nan.shape[1]] &= not_nan
    not_nan = not_nan_ext[radius:-radius,radius:-radius]         
        
    if  not absolute:
        rslt[...,0] +=  rslt[...,1]
    nn_disparity =     np.nan_to_num(rslt[...,0], copy = False)
    target_disparity = np.nan_to_num(rslt[...,1], copy = False)
    gt_disparity =     np.nan_to_num(rslt[...,2], copy = False)
    gt_strength =      np.nan_to_num(rslt[...,3], copy = False)
    rslt = []
    print ("--------------- %s ---------------"%(rslt_path))
    if logfile:
        print ("--------------- %s ---------------"%(rslt_path), file=logfile)

    for min_disparity, max_disparity, max_offset_target, max_offset_result, strength_pow in variants:
        good_tiles = not_nan.copy();
        good_tiles &= (gt_disparity >= min_disparity)
        good_tiles &= (gt_disparity <= max_disparity)
        good_tiles &= (target_disparity != gt_disparity)
        good_tiles &= (np.abs(target_disparity - gt_disparity) <= max_offset_target)
        good_tiles &= (np.abs(target_disparity - nn_disparity) <= max_offset_result)
        gt_w =  gt_strength * good_tiles
        gt_w = np.power(gt_w,strength_pow)
        sw = gt_w.sum() 
        diff0 = target_disparity - gt_disparity
        diff1 = nn_disparity -     gt_disparity
        diff0_2w = gt_w*diff0*diff0
        diff1_2w = gt_w*diff1*diff1
        rms0 = np.sqrt(diff0_2w.sum()/sw) 
        rms1 = np.sqrt(diff1_2w.sum()/sw)
        print ("%7.3f<disp<%7.3f, offs_tgt<%5.2f, offs_rslt<%5.2f pwr=%05.3f, rms0=%7.4f, rms1=%7.4f (gain=%7.4f) num good tiles = %5d"%(
            min_disparity, max_disparity, max_offset_target,  max_offset_result, strength_pow, rms0, rms1, rms0/rms1, good_tiles.sum() ))
        if logfile:
            print ("%7.3f<disp<%7.3f, offs_tgt<%5.2f, offs_rslt<%5.2f pwr=%05.3f, rms0=%7.4f, rms1=%7.4f (gain=%7.4f) num good tiles = %5d"%(
                min_disparity, max_disparity, max_offset_target,  max_offset_result, strength_pow, rms0, rms1, rms0/rms1, good_tiles.sum() ),file=logfile)
            
        rslt.append([rms0,rms1])
    return rslt 

def concentricSquares(radius):
    side = 2 * radius + 1
    return [[((i // side) >= var) and
             ((i // side) < (side - var)) and
             ((i % side)  >= var) and
             ((i % side)  < (side - var))  for i in range (side*side) ] for var in range(radius+1)]    
                         
