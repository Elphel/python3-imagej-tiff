#!/usr/bin/env python3
from tensorflow.python.framework.ops import GraphKeys
__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "andrey@elphel.com"

#Builds (and saved) inference model from trained by nn_ds_neibs21.py
#Model and weights are used by the inference-only infer_qcds_graph.py
import os
import sys
import numpy as np
import time
import shutil
import qcstereo_network
import qcstereo_functions as qsf
import tensorflow as tf
from tensorflow.python.ops import resource_variable_ops

tf.ResourceVariable = resource_variable_ops.ResourceVariable

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
parameters, dirs, files, _ = qsf.parseXmlConfig(conf_file, root_dir)
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
CLUSTER_RADIUS = None
PARTIALS_WEIGHTS, MAX_IMGS_IN_MEM, MAX_FILES_PER_GROUP,  BATCH_WEIGHTS, ONLY_TILE = [None] * 5  
USE_CONFIDENCE, WBORDERS_ZERO, EPOCHS_TO_RUN, FILE_UPDATE_EPOCHS = [None] * 4
LR600,LR400,LR200,LR100,LR = [None]*5
SHUFFLE_FILES, EPOCHS_FULL_TEST, SAVE_TIFFS = [None] * 3
CHECKPOINT_PERIOD = None
TRAIN_BUFFER_GPU, TRAIN_BUFFER_CPU = [None]*2
TEST_TITLES = None
USE_SPARSE_ONLY = True
LOGFILE="results-infer.txt"
"""
Next gets globals from the config file
"""
globals().update(parameters)
WIDTH =  324
HEIGHT = 242
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
NN_LAYOUT1 = qcstereo_network.NN_LAYOUTS[NET_ARCH1]
NN_LAYOUT2 = qcstereo_network.NN_LAYOUTS[NET_ARCH2]
# Tiff export slice labels
SLICE_LABELS =  ["nn_out_ext","hier_out_ext","gt_disparity","gt_strength"]#,
#                 "cutcorn_cost_nw","cutcorn_cost",
#                 "gt-avg_dist","avg8_disp","gt_disp","out-avg"]
##############################################################################
cluster_size = (2 * CLUSTER_RADIUS + 1) * (2 * CLUSTER_RADIUS + 1)
center_tile_index = 2 * CLUSTER_RADIUS * (CLUSTER_RADIUS + 1)
qsf.prepareFiles(dirs,
                 files,
                 suffix = SUFFIX)

print ("Copying config files to results directory:\n ('%s' -> '%s')"%(conf_file,dirs['result']))
try:
    os.makedirs(dirs['result'])
except:
    pass

shutil.copy2(conf_file,dirs['result'])
LOGPATH = os.path.join(dirs['result'],LOGFILE)





image_data = qsf.initImageData( # just use image_data[0]
                files =          files,
                max_imgs =       MAX_IMGS_IN_MEM,
                cluster_radius = 0, # CLUSTER_RADIUS,
                tile_layers =    TILE_LAYERS,
                tile_side =      TILE_SIDE,
                width =          IMG_WIDTH,
                replace_nans =   True,
                infer =          True,
                keep_gt =        True) # to generate same output files

ph_corr2d =           tf.placeholder(np.float32, (None,FEATURES_PER_TILE), name = 'ph_corr2d')
ph_target_disparity = tf.placeholder(np.float32, (None,1),                 name = 'ph_target_disparity')
ph_ntile =            tf.placeholder(np.int32,   (None,),                  name = 'ph_ntile')  #nTile
ph_ntile_out =        tf.placeholder(np.int32,   (None,),                  name = 'ph_ntile_out')  #which tiles should be calculated in stage2 

#corr2d9x325 = tf.concat([tf.reshape(next_element_tt['corr2d'],[-1,cluster_size,FEATURES_PER_TILE]) , tf.reshape(next_element_tt['target_disparity'], [-1,cluster_size, 1])],2)
tf_intile325 =  tf.concat([ph_corr2d, ph_target_disparity],axis=1,name="tf_intile325") # [?,325]
pass
"""
target_disparity_cluster = tf.reshape(next_element_tt['target_disparity'], [-1,cluster_size, 1], name="targdisp_cluster")    
corr2d_Nx325 = tf.concat([tf.reshape(next_element_tt['corr2d'],[-1,cluster_size,FEATURES_PER_TILE], name="coor2d_cluster"),
                          target_disparity_cluster], axis=2, name = "corr2d_Nx325")
"""
cluster_radius = CLUSTER_RADIUS
"""
Probably ResourceVariable is not needed here because of the tf.scatter_update() 

If collection is not provided, it defaults to  [GraphKeys.GLOBAL_VARIABLES], and that in turn fails saver.restore() as this variable was not available in the trained model
"""

'''
#rv_stage1_out = resource_variable_ops.ResourceVariable(
rv_stage1_out = tf.Variable(
    np.zeros([HEIGHT * WIDTH, NN_LAYOUT1[-1]]),
##    collections = [],
    collections = [GraphKeys.LOCAL_VARIABLES],# Works, available with tf.local_variables()
    dtype=np.float32,
    name = 'rv_stage1_out')
'''

rv_stage1_out = tf.get_variable("rv_stage1_out",
                                shape=[HEIGHT * WIDTH, NN_LAYOUT1[-1]],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer,
                                collections = [GraphKeys.LOCAL_VARIABLES],trainable=False)


#rv_stageX_out_init_placeholder = tf.placeholder(tf.float32, shape=[HEIGHT * WIDTH, NN_LAYOUT1[-1]])
#rv_stageX_out_init_op = rv_stageX_out.assign(rv_stageX_out_init_placeholder)

##stage1_tiled = tf.reshape(rv_stage1_out.read_value(),[HEIGHT, WIDTH, -1], name = 'stage1_tiled') 
stage1_tiled = tf.reshape(rv_stage1_out, [HEIGHT, WIDTH, -1], name = 'stage1_tiled') # no need to synchronize here?

tf_stage1_exth =  tf.concat([stage1_tiled[:,:1,:]]*cluster_radius +
                            [stage1_tiled] +
                            [stage1_tiled[:,-1:,:]]*cluster_radius, axis = 1,name = 'stage1_exth')
tf_stage1_ext =   tf.concat([tf_stage1_exth[ :1,:,:]]*cluster_radius +
                            [tf_stage1_exth] +
                            [tf_stage1_exth[-1:,:,:]]*cluster_radius, axis = 0, name = 'stage1_exth')
tf_stage1_ext4 =  tf.expand_dims(tf_stage1_ext, axis = 2, name = 'stage1_ext4')
concat_list = []
cluster_side = 2 * cluster_radius+1
for dy in range(cluster_side):
    for dx in range(cluster_side):
#        concat_list.append(tf_stage1_ext4[dy: cluster_side-dy, dx: cluster_side-dx,:,:])
        concat_list.append(tf.slice(tf_stage1_ext4,[dy,dx,0,0],[HEIGHT, WIDTH,-1,-1]))
        pass
tf_stage2_inm = tf.concat(concat_list, axis = 2, name ='stage2_inm') #242, 324, 25, 64 
tf_stage2_in = tf.reshape(tf_stage2_inm,[-1,rv_stage1_out.shape[1]*cluster_side*cluster_side], name = 'stage2_in')    

tf_stage2_in_sparse = tf.gather(tf_stage2_in, indices= ph_ntile_out, axis=0, name = 'stage2_in_sparse')

#aextv=np.concatenate([a[:,:1,:]]*2 + [a] + [a[:,-1:,:]]*2,axis = 1)
#ext=np.concatenate([aextv[:1,:,:]]*1 + [aextv] + [aextv[-1:,:,:]]*3,axis = 0)


with tf.name_scope("Disparity_net"): # to have the same scope for weight/biases?
    ns, _ = qcstereo_network.network_sub(tf_intile325,
                                 input_global = [None,ph_target_disparity][SPREAD_CONVERGENCE], # input_global[:,i,:],
                                 layout= NN_LAYOUT1,
                                 reuse= False,
                                 sym8 = SYM8_SUB,
                                 cluster_radius = 0)
    update=tf.scatter_update(ref=rv_stage1_out,
                             indices = ph_ntile,
                             updates = ns,
                             use_locking = False,
                             name = 'update')
    with tf.control_dependencies([update]):
        stage1done = tf.constant(1, dtype=tf.int32, name="stage1done")
        pass
    stage2_out_sparse0 = qcstereo_network.network_inter (
                                                 input_tensor =   tf_stage2_in_sparse,
                                                 input_global =   None, #  [None, ig][inter_convergence], # optionally feed all convergence values (from each tile of a cluster)
                                                 layout =         NN_LAYOUT2,
                                                 reuse =          False,
                                                 use_confidence = False)
    stage2_out_sparse = tf.identity(stage2_out_sparse0, name = 'stage2_out_sparse')
     
    if not USE_SPARSE_ONLY: #Does it reduce the graph size?
        stage2_out_full0 = qcstereo_network.network_inter (
                                                 input_tensor =   tf_stage2_in,
                                                 input_global =   None, #  [None, ig][inter_convergence], # optionally feed all convergence values (from each tile of a cluster)
                                                 layout =         NN_LAYOUT2,
                                                 reuse =          True,
                                                 use_confidence = False)
        stage2_out_full = tf.identity(stage2_out_full0, name = 'stage2_out_full') 

    pass

ROOT_PATH  = './attic/infer_qcds_graph'+SUFFIX+"/" # for tensorboard
"""
This is needed if ResourceVariable is used - then i/o tensors names somehow disappeared
and were replaced by 'Placeholder_*'
collection_io = 'collection_io'
tf.add_to_collection(collection_io, ph_corr2d)
tf.add_to_collection(collection_io, ph_target_disparity)
tf.add_to_collection(collection_io, ph_ntile)
tf.add_to_collection(collection_io, ph_ntile_out)
tf.add_to_collection(collection_io, stage1done)
tf.add_to_collection(collection_io, stage2_out_sparse)
"""
##saver=tf.train.Saver()
saver  =tf.train.Saver(tf.global_variables())
#saver = tf.train.Saver(tf.global_variables()+tf.local_variables())

saver_def = saver.as_saver_def()

pass
"""
saver_def = saver.as_saver_def()

# The name of the tensor you must feed with a filename when saving/restoring.
print ('saver_def.filename_tensor_name=',saver_def.filename_tensor_name)

# The name of the target operation you must run when restoring.
print ('saver_def.restore_op_name=',saver_def.restore_op_name)

# The name of the target operation you must run when saving.
print ('saver_def.save_tensor_name=',saver_def.save_tensor_name)

saver_def.filename_tensor_name= save/Const:0
saver_def.restore_op_name= save/restore_all
saver_def.save_tensor_name= save/control_dependency:0
print(saver.save(sess, files["checkpoints"]))
"""
try:
    os.makedirs(os.path.dirname(files['inference']))
    print ("Created directory ",os.path.dirname(files['inference']))
except:
    pass


with tf.Session()  as sess:
        
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    saver.restore(sess, files["checkpoints"])
    
    '''
        rv_stage1_out belongs to GraphKeys.LOCAL_VARIABLES
        Now when weights/biases are restored from 'checkpoints', 
        that do not have this variable, add it to globals.
        Actually it could have been declared right here - this
        needs testing.
        
        NOTE1: The line below makes the next script's, that saves 
        a Saved_Model MetaGraph, size of the Saved_Model significantly 
        bigger.
        
        NOTE2: The line below  is commented in favor of (in the next script!):
            builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING],main_op=tf.local_variables_initializer()) 
    '''
    #tf.add_to_collection(GraphKeys.GLOBAL_VARIABLES, rv_stage1_out)
    
    
    saver.save(sess, files["inference"])
      
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(ROOT_PATH, sess.graph)
    lf = None
    if LOGPATH:
        lf=open(LOGPATH,"w") #overwrite previous (or make it "a"?

    #_ = sess.run([rv_stageX_out_init_op],feed_dict={rv_stageX_out_init_placeholder: np.zeros((HEIGHT * WIDTH, NN_LAYOUT1[-1]))})

    for nimg,_ in enumerate(image_data):
        dataset_img = qsf.readImageData(
            image_data =     image_data,
            files =          files,
            indx =           nimg,
            cluster_radius = 0, # CLUSTER_RADIUS,
            tile_layers =    TILE_LAYERS,
            tile_side =      TILE_SIDE,
            width =          IMG_WIDTH,
            replace_nans =   True,
            infer =          True,
            keep_gt =        True) # to generate same output files
            
        
        
        img_corr2d = dataset_img['corr2d'] # [?,324)
        img_target = dataset_img['target_disparity'] # [?,324)
        img_ntile =  dataset_img['ntile'].reshape([-1])
        #run first stage network
        qsf.print_time("Running inferred model, stage1", end=" ")
        _  = sess.run([stage1done],
                        feed_dict={ph_corr2d:            img_corr2d,
                                   ph_target_disparity:  img_target,
                                   ph_ntile:             img_ntile })
        qsf.print_time("Done.")
        qsf.print_time("Running inferred model, stage2", end=" ")
        disp_out,  = sess.run([stage2_out_sparse],
                        feed_dict={ph_ntile_out:         img_ntile })
        qsf.print_time("Done.")
        result_file = files['result'][nimg].replace('.npy','-infer.npy') #not to overwrite training result files that are more complete
        try:
            os.makedirs(os.path.dirname(result_file))
        except:
            pass
        
        rslt = np.concatenate(
            [disp_out.reshape(-1,1),
             dataset_img['t_disps'], #t_disps[ntest],
             dataset_img['gtruths'], # gtruths[ntest],
             ],1)
        np.save(result_file,           rslt.reshape(HEIGHT,WIDTH,-1))
        rslt = qsf.eval_results(result_file, ABSOLUTE_DISPARITY, radius=CLUSTER_RADIUS, logfile=lf)  # (re-loads results). Only uses first 4 layers
        if SAVE_TIFFS:
            qsf.result_npy_to_tiff(result_file, ABSOLUTE_DISPARITY, fix_nan = True, labels=SLICE_LABELS, logfile=lf)

        """
        Remove dataset_img (if it is not [0] to reduce memory footprint         
        """
        image_data[nimg] = None
    
    # is this needed? why would it be?
    #meta_graph_def = tf.train.export_meta_graph(files["inference"]+'.meta')
                
    
    if lf:
        lf.close()
    writer.close()
