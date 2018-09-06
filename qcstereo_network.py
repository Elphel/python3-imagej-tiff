#!/usr/bin/env python3
__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "andrey@elphel.com"

#from numpy import float64
#import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


def lrelu(x):
    return tf.maximum(x*0.2,x)
#    return tf.nn.relu(x) 

def sym_inputs8(inp, cluster_radius = 2):
    """
    get input vector [?:4*9*9+1] (last being target_disparity) and reorder for horizontal flip,
    vertical flip and transpose (8 variants, mode + 1 - hor, +2 - vert, +4 - transpose)
    return same lengh, reordered
    """
    tile_side = 2 * cluster_radius + 1
    with tf.name_scope("sym_inputs8"):
        td =           inp[:,-1:] # tf.reshape(inp,[-1], name = "td")[-1]
        inp_corr =     tf.reshape(inp[:,:-1],[-1,4,tile_side,tile_side], name = "inp_corr")
        inp_corr_h =   tf.stack([-inp_corr  [:,0,:,-1::-1], inp_corr  [:,1,:,-1::-1], -inp_corr  [:,3,:,-1::-1], -inp_corr  [:,2,:,-1::-1]], axis=1, name = "inp_corr_h")
        inp_corr_v =   tf.stack([ inp_corr  [:,0,-1::-1,:],-inp_corr  [:,1,-1::-1,:],  inp_corr  [:,3,-1::-1,:],  inp_corr  [:,2,-1::-1,:]], axis=1, name = "inp_corr_v")
        inp_corr_hv =  tf.stack([ inp_corr_h[:,0,-1::-1,:],-inp_corr_h[:,1,-1::-1,:],  inp_corr_h[:,3,-1::-1,:],  inp_corr_h[:,2,-1::-1,:]], axis=1, name = "inp_corr_hv")
        inp_corr_t =   tf.stack([tf.transpose(inp_corr   [:,1], perm=[0,2,1]),
                                 tf.transpose(inp_corr   [:,0], perm=[0,2,1]),
                                 tf.transpose(inp_corr   [:,2], perm=[0,2,1]),
                                -tf.transpose(inp_corr   [:,3], perm=[0,2,1])], axis=1, name = "inp_corr_t")
        inp_corr_ht =  tf.stack([tf.transpose(inp_corr_h [:,1], perm=[0,2,1]),
                                 tf.transpose(inp_corr_h [:,0], perm=[0,2,1]),
                                 tf.transpose(inp_corr_h [:,2], perm=[0,2,1]),
                                -tf.transpose(inp_corr_h [:,3], perm=[0,2,1])], axis=1, name = "inp_corr_ht")
        inp_corr_vt =  tf.stack([tf.transpose(inp_corr_v [:,1], perm=[0,2,1]),
                                 tf.transpose(inp_corr_v [:,0], perm=[0,2,1]),
                                 tf.transpose(inp_corr_v [:,2], perm=[0,2,1]),
                                -tf.transpose(inp_corr_v [:,3], perm=[0,2,1])], axis=1, name = "inp_corr_vt")
        inp_corr_hvt = tf.stack([tf.transpose(inp_corr_hv[:,1], perm=[0,2,1]),
                                 tf.transpose(inp_corr_hv[:,0], perm=[0,2,1]),
                                 tf.transpose(inp_corr_hv[:,2], perm=[0,2,1]),
                                -tf.transpose(inp_corr_hv[:,3], perm=[0,2,1])], axis=1, name = "inp_corr_hvt")
#        return td, [inp_corr, inp_corr_h, inp_corr_v, inp_corr_hv, inp_corr_t, inp_corr_ht, inp_corr_vt, inp_corr_hvt]
        """
        return [tf.concat([tf.reshape(inp_corr,    [inp_corr.shape[0],-1]),td], axis=1,name = "out_corr"),
                tf.concat([tf.reshape(inp_corr_h,  [inp_corr.shape[0],-1]),td], axis=1,name = "out_corr_h"),
                tf.concat([tf.reshape(inp_corr_v,  [inp_corr.shape[0],-1]),td], axis=1,name = "out_corr_v"),
                tf.concat([tf.reshape(inp_corr_hv, [inp_corr.shape[0],-1]),td], axis=1,name = "out_corr_hv"),
                tf.concat([tf.reshape(inp_corr_t,  [inp_corr.shape[0],-1]),td], axis=1,name = "out_corr_t"),
                tf.concat([tf.reshape(inp_corr_ht, [inp_corr.shape[0],-1]),td], axis=1,name = "out_corr_ht"),
                tf.concat([tf.reshape(inp_corr_vt, [inp_corr.shape[0],-1]),td], axis=1,name = "out_corr_vt"),
                tf.concat([tf.reshape(inp_corr_hvt,[inp_corr.shape[0],-1]),td], axis=1,name = "out_corr_hvt")]
        """
        cl = 4 * tile_side * tile_side
        return [tf.concat([tf.reshape(inp_corr,    [-1,cl]),td], axis=1,name = "out_corr"),
                tf.concat([tf.reshape(inp_corr_h,  [-1,cl]),td], axis=1,name = "out_corr_h"),
                tf.concat([tf.reshape(inp_corr_v,  [-1,cl]),td], axis=1,name = "out_corr_v"),
                tf.concat([tf.reshape(inp_corr_hv, [-1,cl]),td], axis=1,name = "out_corr_hv"),
                tf.concat([tf.reshape(inp_corr_t,  [-1,cl]),td], axis=1,name = "out_corr_t"),
                tf.concat([tf.reshape(inp_corr_ht, [-1,cl]),td], axis=1,name = "out_corr_ht"),
                tf.concat([tf.reshape(inp_corr_vt, [-1,cl]),td], axis=1,name = "out_corr_vt"),
                tf.concat([tf.reshape(inp_corr_hvt,[-1,cl]),td], axis=1,name = "out_corr_hvt")]
#                           inp_corr_h, inp_corr_v, inp_corr_hv, inp_corr_t, inp_corr_ht, inp_corr_vt, inp_corr_hvt]
    

def network_sub(input_tensor,
                input_global,  #add to all layers (but first) if not None
                layout,
                reuse,
                sym8 = False,
                cluster_radius = 2):
#    last_indx = None;
    fc = []
    inp_weights = []
    for i, num_outs in enumerate (layout):
        if num_outs:
            if fc:
                if input_global is None:
                    inp = fc[-1]
                else:
                    inp = tf.concat([fc[-1], input_global], axis = 1)
                fc.append(slim.fully_connected(inp,    num_outs, activation_fn=lrelu, scope='g_fc_sub'+str(i), reuse = reuse))
            else:
                inp = input_tensor
                if sym8:
                    inp8 = sym_inputs8(inp, cluster_radius)
                    num_non_sum = num_outs %  len(inp8) # if number of first layer outputs is not multiple of 8
                    num_sym8 =    num_outs // len(inp8) # number of symmetrical groups
                    fc_sym = []
                    for j in range (len(inp8)): # ==8
                        reuse_this = reuse | (j > 0)
                        scp = 'g_fc_sub'+str(i)
                        fc_sym.append(slim.fully_connected(inp8[j],    num_sym8, activation_fn=lrelu, scope= scp,     reuse = reuse_this))
                        if not reuse_this:
                            with tf.variable_scope(scp,reuse=True) : # tf.AUTO_REUSE):
                                inp_weights.append(tf.get_variable('weights')) # ,shape=[inp.shape[1],num_outs])) 
                    if num_non_sum > 0:
                        reuse_this = reuse
                        scp = 'g_fc_sub'+str(i)+"r"
                        fc_sym.append(slim.fully_connected(inp,     num_non_sum, activation_fn=lrelu, scope=scp, reuse = reuse_this))    
                        if not reuse_this:
                            with tf.variable_scope(scp,reuse=True) : # tf.AUTO_REUSE):
                                inp_weights.append(tf.get_variable('weights')) # ,shape=[inp.shape[1],num_outs])) 
                    fc.append(tf.concat(fc_sym, 1, name='sym_input_layer'))
                else:
                    scp = 'g_fc_sub'+str(i)
                    fc.append(slim.fully_connected(inp,    num_outs, activation_fn=lrelu, scope= scp, reuse = reuse))
                    if not reuse:
                        with tf.variable_scope(scp, reuse=True) : # tf.AUTO_REUSE):
                            inp_weights.append(tf.get_variable('weights')) # ,shape=[inp.shape[1],num_outs])) 
           
    return fc[-1], inp_weights

def network_inter(input_tensor,
                  input_global,  #add to all layers (but first) if not None
                  layout,
                  reuse=False,
                  use_confidence=False):
    #last_indx = None;
    fc = []
    for i, num_outs in enumerate (layout):
        if num_outs:
            if fc:
                if input_global is None:
                    inp = fc[-1]
                else:
                    inp = tf.concat([fc[-1], input_global], axis = 1)
            else:
                inp = input_tensor
            fc.append(slim.fully_connected(inp,    num_outs, activation_fn=lrelu, scope='g_fc_inter'+str(i), reuse = reuse))
    if use_confidence:
        fc_out  = slim.fully_connected(fc[-1],     2, activation_fn=lrelu, scope='g_fc_inter_out', reuse = reuse)
    else:     
        fc_out  = slim.fully_connected(fc[-1],     1, activation_fn=None, scope='g_fc_inter_out', reuse = reuse)
        #If using residual disparity, split last layer into 2 or remove activation and add rectifier to confidence only  
    return fc_out

def networks_siam(input_tensor, # now [?,9,325]-> [?,25,325]
                  input_global, # add to all layers (but first) if not None
                  layout1, 
                  layout2,
                  inter_convergence,
                  sym8 =        False,
                  only_tile =   None, # just for debugging - feed only data from the center sub-network
                  partials =    None,
                  use_confidence=False,
                  cluster_radius = 2):
                  
    center_index = (input_tensor.shape[1] - 1) // 2 
    with tf.name_scope("Siam_net"):
        inp_weights = []
        num_legs =  input_tensor.shape[1] # == 25
        if partials is None:
            partials = [[True] * num_legs]
        inter_lists = [[] for _ in partials]
        reuse = False
        for i in range (num_legs):
            if ((only_tile is None) or (i == only_tile)) and any([p[i] for p in partials]) :
                if input_global is None:
                    ig = None
                else:
                    ig =input_global[:,i,:]
                ns, ns_weights = network_sub(input_tensor[:,i,:],
                                             ig, # input_global[:,i,:],
                                             layout= layout1,
                                             reuse= reuse,
                                             sym8 = sym8,
                                             cluster_radius = cluster_radius)
                for n, partial in enumerate(partials):
                    if partial[i]:
                        inter_lists[n].append(ns)
                    else:
                        inter_lists[n].append(tf.zeros_like(ns))
                inp_weights += ns_weights
                reuse = True
        outs = []         
        for n, _ in enumerate(partials):
            if input_global is None:
                ig = None
            else:
                ig =input_global[:,center_index,:]
            
            outs.append(network_inter (input_tensor = tf.concat(inter_lists[n],
                                                 axis=1,
                                                 name='inter_tensor'+str(n)),
                                       input_global =   [None, ig][inter_convergence], # optionally feed all convergence values (from each tile of a cluster)
                                       layout =         layout2,
                                       reuse =          (n > 0),
                                       use_confidence = use_confidence))
        return  outs,  inp_weights 



