#!/usr/bin/env python3
__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "andrey@elphel.com"

# Just inference, currently uses /data_ssd/data_sets/tf_data_5x5_main_13_heur/inference/
import os
import sys
import numpy as np
import time
import shutil
##import qcstereo_network
import qcstereo_functions as qsf
import tensorflow as tf
#from tensorflow.python.ops import resource_variable_ops
#tf.ResourceVariable = resource_variable_ops.ResourceVariable

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
##NN_LAYOUT1 = qcstereo_network.NN_LAYOUTS[NET_ARCH1]
##NN_LAYOUT2 = qcstereo_network.NN_LAYOUTS[NET_ARCH2]
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

"""
Next is tag for pb (pb == protocol buffer) model
"""
#PB_TAGS = ["model_pb"]

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

cluster_radius = CLUSTER_RADIUS

ROOT_PATH  = './attic/infer_qcds_graph'+SUFFIX+"/" # for tensorboard

try:
    os.makedirs(os.path.dirname(files['inference']))
    print ("Created directory ",os.path.dirname(files['inference']))
except:
    pass

with tf.Session()  as sess:
    
    # Actually, refresh all the time and have an extra script to restore from it.  
    # use_Saved_Model = False
    
    #if os.path.isdir(dirs['exportdir']):
    #    # check if dir contains "Saved Model" model
    #    use_saved_model = tf.saved_model.loader.maybe_saved_model_directory(dirs['exportdir'])

    #if use_saved_model:
    #    print("Model restore: using Saved_Model model MetaGraph protocol buffer")
    #    meta_graph_source = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], dirs['exportdir'])
    #else:
    
    meta_graph_source = files["inference"]+'.meta'
    
    print("Model restore: using conventionally saved model, but saving Saved Model for the next run")
    print("MetaGraph source = "+str(meta_graph_source))

    infer_saver = tf.train.import_meta_graph(meta_graph_source)
    
    graph=tf.get_default_graph()
    
    ph_corr2d =           graph.get_tensor_by_name('ph_corr2d:0') 
    ph_target_disparity = graph.get_tensor_by_name('ph_target_disparity:0')
    ph_ntile =            graph.get_tensor_by_name('ph_ntile:0')
    ph_ntile_out =        graph.get_tensor_by_name('ph_ntile_out:0')
    
    stage1done =          graph.get_tensor_by_name('Disparity_net/stage1done:0') #<tf.Operation 'Siam_net/stage1done' type=Const>, 
    stage2_out_sparse =   graph.get_tensor_by_name('Disparity_net/stage2_out_sparse:0')#not found
    if not USE_SPARSE_ONLY: #Does it reduce the graph size?
        stage2_out_full = graph.get_tensor_by_name('Disparity_net/stage2_out_full:0') 


    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())


    infer_saver.restore(sess, files["inference"])
    
    
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(ROOT_PATH, sess.graph)
    
    lf = None
    if LOGPATH:
        lf=open(LOGPATH,"w") #overwrite previous (or make it "a"?

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
        img_corr2d = dataset_img['corr2d'] # (?,324)
        img_target = dataset_img['target_disparity'] # (?,1)
        img_ntile =  dataset_img['ntile'].reshape([-1]) # (?) - 0...78k int32
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
            qsf.result_npy_to_tiff(result_file, ABSOLUTE_DISPARITY, fix_nan = True,labels=SLICE_LABELS, logfile=lf)
        """
        Remove dataset_img (if it is not [0] to reduce memory footprint         
        """
        image_data[nimg] = None
    
    #builder.add_meta_graph_and_variables(sess,PB_TAGS)
    # clean
    shutil.rmtree(dirs['exportdir'], ignore_errors=True)
    # save MetaGraph to Saved_Model as *.pb
    builder = tf.saved_model.builder.SavedModelBuilder(dirs['exportdir'])
    builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING],main_op=tf.local_variables_initializer())
    #builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING])
    #builder.save(True)
    builder.save(False) # True = *.pbtxt, False = *.pb
    
    if lf:
        lf.close()
    writer.close()
