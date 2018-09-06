#!/usr/bin/env python3
__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "andrey@elphel.com"

#from numpy import float64
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
    
def parseXmlConfig(conf_file, root_dir):
    tree = ET.parse(conf_file)
    root = tree.getroot()
    parameters = {}
    for p in root.find('parameters'):
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
#    globals().update(parameters)
    dbg_parameters = {}
    for p in root.find('dbg_parameters'):
        dbg_parameters[p.tag]=eval(p.text.strip())

    return parameters, dirs, files, dbg_parameters



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
    result_files=[]
    for i, path in enumerate(files['images']):
        result_files.append(os.path.join(dirs['result'], path+"_"+suffix+'.npy'))
    files['result'] = result_files 
    files['train'] = [files['train_lvar'],files['train_hvar'], files['train_lvar1'], files['train_hvar1']]
    # should be after result files
    for i, path in enumerate(files['images']):
        files['images'][i] =      os.path.join(dirs['images'],    path+'.tfrecords')

def readTFRewcordsEpoch(train_filename):
    if not  '.tfrecords' in train_filename:
        train_filename += '.tfrecords'
    npy_dir_name = "npy"
    dirname = os.path.dirname(train_filename) 
    npy_dir = os.path.join(dirname, npy_dir_name)
#    filebasename, file_extension = os.path.splitext(train_filename)
    filebasename, _ = os.path.splitext(train_filename)
    filebasename = os.path.basename(filebasename)
    file_corr2d =           os.path.join(npy_dir,filebasename + '_corr2d.npy')
    file_target_disparity = os.path.join(npy_dir,filebasename + '_target_disparity.npy')
    file_gt_ds =            os.path.join(npy_dir,filebasename + '_gt_ds.npy')
    if (os.path.exists(file_corr2d) and
        os.path.exists(file_target_disparity) and
        os.path.exists(file_gt_ds)):
        corr2d=            np.load (file_corr2d)
        target_disparity = np.load(file_target_disparity)
        gt_ds =            np.load(file_gt_ds)
        pass
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
            os.makedirs(os.path.dirname(file_corr2d))
        except:
            pass     

        np.save(file_corr2d,           corr2d)
        np.save(file_target_disparity, target_disparity)
        np.save(file_gt_ds,            gt_ds)
    return corr2d, target_disparity, gt_ds

def getMoreFiles(fpaths,rslt, cluster_radius, hor_flip, tile_layers, tile_side):
    for fpath in fpaths:
        corr2d, target_disparity, gt_ds = readTFRewcordsEpoch(fpath)
        dataset = {"corr2d":           corr2d,
                     "target_disparity": target_disparity,
                     "gt_ds":            gt_ds}
        """
        if FILE_TILE_SIDE > TILE_SIDE:
            reduce_tile_size([dataset],   TILE_LAYERS, TILE_SIDE)
        """
        reformat_to_clusters([dataset], cluster_radius)
        if hor_flip:
            if np.random.randint(2):
                print_time("Performing horizontal flip", end=" ")
                flip_horizontal([dataset], cluster_radius, tile_layers, tile_side)
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
    # still some nan-s in correlation data?
#    in_features_clean = tf.where(tf.is_nan(in_features), tf.zeros_like(in_features), in_features)     
#    corr2d_out, target_disparity_out, gt_ds_out = tf.train.shuffle_batch( [in_features_clean, target_disparity, gt_ds],
    corr2d_out, target_disparity_out, gt_ds_out = tf.train.shuffle_batch( [in_features, target_disparity, gt_ds],
                                                 batch_size=1000, # 2,
                                                 capacity=30,
                                                 num_threads=2,
                                                 min_after_dequeue=10)
    return corr2d_out, target_disparity_out, gt_ds_out
#http://adventuresinmachinelearning.com/introduction-tensorflow-queuing/
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

def reformat_to_clusters(datasets_data, cluster_radius):
    cluster_size = (2 * cluster_radius + 1) * (2 * cluster_radius + 1)
# Reformat input data
    for rec in datasets_data:
        rec['corr2d'] =           rec['corr2d'].reshape(          (rec['corr2d'].shape[0]//cluster_size,           rec['corr2d'].shape[1] * cluster_size)) 
        rec['target_disparity'] = rec['target_disparity'].reshape((rec['target_disparity'].shape[0]//cluster_size, rec['target_disparity'].shape[1] * cluster_size)) 
        rec['gt_ds'] =            rec['gt_ds'].reshape(           (rec['gt_ds'].shape[0]//cluster_size,            rec['gt_ds'].shape[1] * cluster_size))

def flip_horizontal(datasets_data, cluster_radius, tile_layers, tile_side):
    cluster_side = 2 * cluster_radius + 1
#    cluster_size = cluster_side * cluster_side
    """
TILE_LAYERS =        4
TILE_SIDE =          9 # 7
TILE_SIZE =         TILE_SIDE* TILE_SIDE # == 81
    """
    for rec in datasets_data:
        corr2d =           rec['corr2d'].reshape(          (rec['corr2d'].shape[0],  cluster_side, cluster_side, tile_layers, tile_side, tile_side))
        target_disparity = rec['target_disparity'].reshape((rec['corr2d'].shape[0],  cluster_side, cluster_side, -1))
        gt_ds =            rec['gt_ds'].reshape(           (rec['corr2d'].shape[0],  cluster_side, cluster_side, -1))
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
        
        
        rec['corr2d'] =           corr2d.reshape((corr2d.shape[0],-1)) 
        rec['target_disparity'] = target_disparity.reshape((target_disparity.shape[0],-1)) 
        rec['gt_ds'] =            gt_ds.reshape((gt_ds.shape[0],-1))

def replace_nan(datasets_data): # , cluster_radius):
#    cluster_size = (2 * cluster_radius + 1) * (2 * cluster_radius + 1)
# Reformat input data
    for rec in datasets_data:
        if not rec is None:
            np.nan_to_num(rec['corr2d'],           copy = False) 
            np.nan_to_num(rec['target_disparity'], copy = False) 
            np.nan_to_num(rec['gt_ds'],            copy = False)

def permute_to_swaps(perm):
    pairs = []
    for i in range(len(perm)):
        w = np.where(perm == i)[0][0]
        if w != i:
            pairs.append([i,w])
            perm[w] = perm[i]
            perm[i] = i
    return pairs        
        
def shuffle_in_place(datasets_data, indx, period):
    swaps = permute_to_swaps(np.random.permutation(len(datasets_data)))
#    num_entries = datasets_data[0]['corr2d'].shape[0] // period
    for swp in swaps:
        ds0 = datasets_data[swp[0]]
        ds1 = datasets_data[swp[1]]
        tmp = ds0['corr2d'][indx::period].copy() 
        ds0['corr2d'][indx::period] = ds1['corr2d'][indx::period]
        ds1['corr2d'][indx::period] = tmp

        tmp = ds0['target_disparity'][indx::period].copy() 
        ds0['target_disparity'][indx::period] = ds1['target_disparity'][indx::period]
        ds1['target_disparity'][indx::period] = tmp

        tmp = ds0['gt_ds'][indx::period].copy() 
        ds0['gt_ds'][indx::period] = ds1['gt_ds'][indx::period]
        ds1['gt_ds'][indx::period] = tmp
    
def shuffle_chunks_in_place(datasets_data, tiles_groups_per_chunk):
    """
    Improve shuffling by preserving indices inside batches (0 <->0, ... 39 <->39 for 40 tile group batches)
    """
#    num_files = len(datasets_data)
    #chunks_per_file = datasets_data[0]['target_disparity']
#    for nf, ds in enumerate(datasets_data):
    for ds in datasets_data:
        groups_per_file = ds['corr2d'].shape[0]
        chunks_per_file = groups_per_file//tiles_groups_per_chunk
        permut = np.random.permutation(chunks_per_file)
        ds['corr2d'] =           ds['corr2d'].          reshape((chunks_per_file,-1))[permut].reshape((groups_per_file,-1))
        ds['target_disparity'] = ds['target_disparity'].reshape((chunks_per_file,-1))[permut].reshape((groups_per_file,-1))
        ds['gt_ds'] =            ds['gt_ds'].           reshape((chunks_per_file,-1))[permut].reshape((groups_per_file,-1))

def _setFileSlot(train_next, files, max_files_per_group):
    train_next['files'] = files
    train_next['slots'] = min(train_next['files'], max_files_per_group)
     
def _nextFileSlot(train_next):
    train_next['file'] = (train_next['file'] + 1) % train_next['files'] 
    train_next['slot'] = (train_next['slot'] + 1) % train_next['slots'] 

    
def replaceNextDataset(datasets_data, new_dataset, train_next, nset,period):
    replaceDataset(datasets_data, new_dataset, nset, period, findx = train_next['slot'])
#    _nextFileSlot(train_next[nset])

        
def replaceDataset(datasets_data, new_dataset, nset, period, findx):
    """
    Replace one file in the dataset
    """
    datasets_data[findx]['corr2d']          [nset::period] =  new_dataset['corr2d'] 
    datasets_data[findx]['target_disparity'][nset::period] =  new_dataset['target_disparity'] 
    datasets_data[findx]['gt_ds']           [nset::period] =  new_dataset['gt_ds'] 
    

def zip_lvar_hvar(datasets_all_data, del_src = True):
#    cluster_size = (2 * CLUSTER_RADIUS + 1) * (2 * CLUSTER_RADIUS + 1)
# Reformat input data
    num_sets_to_combine = len(datasets_all_data)
    datasets_data = []
    if num_sets_to_combine:
        for nrec in range(len(datasets_all_data[0])):
            recs = [[] for _ in range(num_sets_to_combine)]
            for nset, datasets in enumerate(datasets_all_data):
                recs[nset] = datasets[nrec]
                
            rec = {'corr2d':           np.empty((recs[0]['corr2d'].shape[0]*num_sets_to_combine,          recs[0]['corr2d'].shape[1]),dtype=np.float32),
                   'target_disparity': np.empty((recs[0]['target_disparity'].shape[0]*num_sets_to_combine,recs[0]['target_disparity'].shape[1]),dtype=np.float32),
                   'gt_ds':            np.empty((recs[0]['gt_ds'].shape[0]*num_sets_to_combine,           recs[0]['gt_ds'].shape[1]),dtype=np.float32)}
            
#            for nset, reci in enumerate(recs):
            for nset, _ in enumerate(recs):
                rec['corr2d']          [nset::num_sets_to_combine] =  recs[nset]['corr2d'] 
                rec['target_disparity'][nset::num_sets_to_combine] =  recs[nset]['target_disparity'] 
                rec['gt_ds']           [nset::num_sets_to_combine] =  recs[nset]['gt_ds'] 
            if del_src:
                for nset in range(num_sets_to_combine):
                    datasets_all_data[nset][nrec] = None
            datasets_data.append(rec)
    return datasets_data    
         

# list of dictionaries  
def reduce_tile_size(datasets_data, num_tile_layers, reduced_tile_side):
    if (not datasets_data is None) and (len (datasets_data) > 0): 
        tsz = (datasets_data[0]['corr2d'].shape[1])// num_tile_layers # 81 # list index out of range
        tss = int(np.sqrt(tsz)+0.5)
        offs = (tss - reduced_tile_side) // 2
        for rec in datasets_data:
            rec['corr2d'] =  (rec['corr2d'].reshape((-1, num_tile_layers,  tss, tss))
                                 [..., offs:offs+reduced_tile_side, offs:offs+reduced_tile_side].
                                 reshape(-1,num_tile_layers*reduced_tile_side*reduced_tile_side))
            
            
def initTrainTestData(
        files,
        cluster_radius,
        max_files_per_group, # shuffling buffer for files
        two_trains,
        train_next):
#    datasets_train_lvar =  []
#    datasets_train_hvar =  []
#    datasets_train_lvar1 = []
#    datasets_train_hvar1 = []
    datasets_train_all = [[],[],[],[]]
    for n_train, f_train in enumerate(files['train']):
        if len(f_train) and ((n_train<2) or two_trains):
            _setFileSlot(train_next[n_train], len(f_train), max_files_per_group)
            for i, fpath in enumerate(f_train):
                if i >= max_files_per_group:
                    break
                print_time("Importing train data "+(["low variance","high variance", "low variance1","high variance1"][n_train]) +" from "+fpath, end="")
                corr2d, target_disparity, gt_ds = readTFRewcordsEpoch(fpath)
                datasets_train_all[n_train].append({"corr2d":corr2d,
                                            "target_disparity":target_disparity,
                                            "gt_ds":gt_ds})
                _nextFileSlot(train_next[n_train])
                print_time("  Done")
    
    datasets_test_lvar = []
    for fpath in files['test_lvar']:
        print_time("Importing test data (low variance) from "+fpath, end="")
        corr2d, target_disparity, gt_ds = readTFRewcordsEpoch(fpath)
        datasets_test_lvar.append({"corr2d":corr2d,
                                    "target_disparity":target_disparity,
                                    "gt_ds":gt_ds})
        print_time("  Done")
    datasets_test_hvar = []
    for fpath in files['test_hvar']:
        print_time("Importing test data (high variance) from "+fpath, end="")
        corr2d, target_disparity, gt_ds = readTFRewcordsEpoch(fpath)
        datasets_test_hvar.append({"corr2d":corr2d,
                                    "target_disparity":target_disparity,
                                    "gt_ds":gt_ds})
        print_time("  Done")
        
    # Reformat to 1/9/25 tile clusters
    for n_train, d_train in enumerate(datasets_train_all):
        print_time("Reshaping train data ("+(["low variance","high variance", "low variance1","high variance1"][n_train])+") ", end="")
        reformat_to_clusters(d_train, cluster_radius)
        print_time("  Done")
    
    print_time("Reshaping test data (low variance)", end="")
    reformat_to_clusters(datasets_test_lvar, cluster_radius)
    print_time("  Done")
    print_time("Reshaping test data (high variance)", end="")
    reformat_to_clusters(datasets_test_hvar, cluster_radius)
    print_time("  Done")
    pass    
    
    """
    datasets_train_lvar & datasets_train_hvar ( that will increase batch size and placeholders twice
    test has to have even original, batches will not zip - just use two batches for one big one
    """
    print_time("Zipping together datasets datasets_train_lvar and datasets_train_hvar", end="")
    datasets_train = zip_lvar_hvar(datasets_train_all, del_src = True) # no shuffle, delete src
    print_time("  Done")
    
    datasets_test = []
    for dataset_test_lvar in datasets_test_lvar:
        datasets_test.append(dataset_test_lvar)
    for dataset_test_hvar in datasets_test_hvar:
        datasets_test.append(dataset_test_hvar)
    
    return datasets_train, datasets_test, len(datasets_train_all) # 4
    
                
                
                
    
def readImageData(image_data,
                  files,
                  indx,
                  cluster_radius,
                  width,
                  replace_nans):
    if image_data[indx] is None:
        corr2d, target_disparity, gt_ds = readTFRewcordsEpoch(files['images'][indx])
        image_data[indx] = {
            'corr2d':           corr2d,
            'target_disparity': target_disparity,
             "gt_ds":            gt_ds,
             "gtruths":          gt_ds.copy(),
             "t_disps":          target_disparity.reshape([-1,1]).copy()}
        extend_img_to_clusters(
             [image_data[indx]],
             cluster_radius,
             width)
        if replace_nans:
#            replace_nan([image_data[indx]], cluster_radius)
            replace_nan([image_data[indx]])
            
    return image_data[indx]
               
def initImageData(files,
                  max_imgs,
                  cluster_radius,
                  width,
                  replace_nans):
    num_imgs = len(files['images'])
    img_data = [None] * num_imgs
    for nfile in range(min(num_imgs, max_imgs)):
        print_time("Importing test image data from "+ files['images'][nfile], end="")
        readImageData(img_data,files, nfile, cluster_radius, width, replace_nans)
        print_time("  Done")
        return img_data    
            
def evaluateAllResults(result_files, absolute_disparity, cluster_radius):
    for result_file in result_files:
        try:
            print_time("Reading resuts from "+result_file, end=" ")
            eval_results(result_file, absolute_disparity, radius=cluster_radius)
        except:
            print_time(" - does not exist")
            continue
        print_time("Done")
        print_time("Saving resuts to tiff", end=" ")
        result_npy_to_tiff(result_file, absolute_disparity, fix_nan = True)        
        print_time("Done")
    


def result_npy_prepare(npy_path, absolute, fix_nan, insert_deltas=True):
    
    """
    @param npy_path full path to the npy file with 4-layer data (242,324,4) - nn_disparity(offset), target_disparity, gt disparity, gt strength
           data will be written as 4-layer tiff, extension '.npy' replaced with '.tiff'
    @param absolute - True - the first layer contains absolute disparity, False - difference from target_disparity
    @param fix_nan - replace nan in target_disparity with 0 to apply offset, target_disparity will still contain nan
    """
    data = np.load(npy_path) #(324,242,4) [nn_disp, target_disp,gt_disp, gt_conf]
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
        data = np.concatenate([data[...,0:4],data[...,0:2],data[...,0:2],data[...,4:]], axis = 2)
        data[...,6] -= data[...,gt_disparity] 
        data[...,7] -= data[...,gt_disparity]
        for l in [2, 4, 5, 6, 7]:
            data[...,l] = np.select([data[...,gt_strength]==0.0, data[...,gt_strength]>0.0], [np.nan,data[...,l]])
        # All other layers - mast too
        for l in range(8,data.shape[2]):
            data[...,l] = np.select([data[...,gt_strength]==0.0, data[...,gt_strength]>0.0], [np.nan,data[...,l]])
    return data        

def result_npy_to_tiff(npy_path, absolute, fix_nan, insert_deltas=True):
    
    """
    @param npy_path full path to the npy file with 4-layer data (242,324,4) - nn_disparity(offset), target_disparity, gt disparity, gt strength
           data will be written as 4-layer tiff, extension '.npy' replaced with '.tiff'
    @param absolute - True - the first layer contains absolute disparity, False - difference from target_disparity
    @param fix_nan - replace nan in target_disparity with 0 to apply offset, target_disparity will still contain nan
    """
    data = result_npy_prepare(npy_path, absolute, fix_nan, insert_deltas)
    tiff_path = npy_path.replace('.npy','.tiff')
            
    data = data.transpose(2,0,1)
    print("Saving results to TIFF: "+tiff_path)
    imagej_tiffwriter.save(tiff_path,data[...,np.newaxis])        

def eval_results(rslt_path, absolute,
                 min_disp =       -0.1, #minimal GT disparity
                 max_disp =       20.0, # maximal GT disparity
                 max_ofst_target = 1.0,
                 max_ofst_result = 1.0,
                 str_pow =         2.0,
                 radius =          0):
#    for min_disparity, max_disparity, max_offset_target, max_offset_result, strength_pow in [
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
        rslt.append([rms0,rms1])
    return rslt 

def concentricSquares(radius):
    side = 2 * radius + 1
    return [[((i // side) >= var) and
             ((i // side) < (side - var)) and
             ((i % side)  >= var) and
             ((i % side)  < (side - var))  for i in range (side*side) ] for var in range(radius+1)]    
                         
