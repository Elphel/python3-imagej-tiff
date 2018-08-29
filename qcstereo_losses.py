#!/usr/bin/env python3
__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "andrey@elphel.com"

#from numpy import float64
import numpy as np
import tensorflow as tf

def smoothLoss(out_batch,                   # [batch_size,(1..2)] tf_result
               target_disparity_batch,      # [batch_size]        tf placeholder
               gt_ds_batch_clust,           # [batch_size,25,2]      tf placeholder
               absolute_disparity =     False, #when false there should be no activation on disparity output !
               cluster_radius =         2):
    with tf.name_scope("SmoothLoss"):
        center_tile_index = 2 * cluster_radius * (cluster_radius + 1)
        cluster_side =      2 * cluster_radius + 1
        cluster_size = cluster_side * cluster_side
        w_corner = 0.7
        w8 = [w_corner,1.0,w_corner,1.0,1.0,w_corner,1.0,w_corner]
        w8 = [w/sum(w8) for w in w8]   
        tf_w8=tf.reshape(tf.constant(w8, dtype=tf.float32, name="w8_"), shape=[1,-1], name="w8")
        i8 = []
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                if (dy != 0) or (dx != 0):
                    i8.append(center_tile_index+(dy*cluster_side)+dx)
        tf_gt_ds_all =     tf.reshape(gt_ds_batch_clust,[-1,cluster_size,gt_ds_batch_clust.shape[1]//cluster_size], name = "gt_ds_all")            
        tf_neibs8 =        tf.gather(tf_gt_ds_all, indices = i8, axis = 1,       name = "neibs8")
        tf_gt_disparity8 = tf.reshape(tf_neibs8[:,:,0], [-1,8],                       name = "gt8_disparity") # (?,8)
        tf_gt_strength8 =  tf.reshape(tf_neibs8[:,:,1], [-1,8],                       name = "gt8_strength") # (?,8)
        tf_w =             tf.multiply(tf_gt_strength8,  tf_w8,                       name = "w")
        tf_dw =            tf.multiply(tf_gt_disparity8, tf_w,                        name = "dw")
        tf_sum_w =         tf.reduce_sum(tf_w,  axis = 1,                             name = "sum_w")
        tf_sum_dw =        tf.reduce_sum(tf_dw, axis = 1,                             name = "sum_dw")
        tf_avg_disparity = tf.divide(tf_sum_dw, tf_sum_w,                             name = "avg_disparity") # (?,)
        tf_gt_disparity =  tf.reshape(tf_gt_ds_all[:,center_tile_index,0], [-1], name = "gt_disparity") # (?,)
        """
        It is good to limit tf_gt_disparityby min/max (+margin) tf.reduce_min(tf_gt_disparity8, axis=1,...) but there could be zeros caused by undefined GT for the tile
        """

        tf_gt_strength =   tf.reshape(tf_gt_ds_all[:,center_tile_index,1], [-1], name = "gt_strength") # (?,)
        tf_d0 =            tf.abs(tf_gt_disparity - tf_avg_disparity,                 name = "tf_d0")
        tf_d =             tf.maximum(tf_d0, 0.001,                                   name = "tf_d")
        tf_d2 =            tf.multiply(tf_d, tf_d,                                    name = "tf_d2")
        
        tf_out =           tf.reshape(out_batch[:,0],[-1],                            name = "tf_out")
        if absolute_disparity:
            tf_out_disparity = tf_out
        else:
            tf_out_disparity = tf.add(tf_out, tf.reshape(target_disparity_batch,[-1]),name = "out_disparity")
            
        tf_offs =          tf.subtract(tf_out_disparity, tf_avg_disparity,            name = "offs")
        tf_offs2 =         tf.multiply(tf_offs, tf_offs,                              name = "offs2")
#        tf_parab =         tf.divide(tf_offs2, tf_d,                                  name = "parab")
#        tf_cost_nlim =     tf.subtract(tf_d2, tf_offs2,                               name = "cost_nlim")

    
#        tf_cost_nw =       tf.maximum(tf_d - tf_parab, 0.0,                           name = "cost_nw")
        tf_cost_nw =       tf.maximum(tf_d2 - tf_offs2, 0.0,                          name = "cost_nw")
        tf_cost_w =        tf.multiply(tf_cost_nw, tf_gt_strength,                    name = "cost_w")
        tf_sum_wc =        tf.reduce_sum(tf_gt_strength,                              name = "sum_wc")
        tf_sum_costw =     tf.reduce_sum(tf_cost_w,                                   name = "sum_costw")
        tf_cost =          tf.divide(tf_sum_costw, tf_sum_wc,                         name = "cost")
        return tf_cost, tf_cost_nw, tf_cost_w, tf_d , tf_avg_disparity, tf_gt_disparity, tf_offs


def batchLoss(out_batch,                   # [batch_size,(1..2)] tf_result
              target_disparity_batch,      # [batch_size]        tf placeholder
              gt_ds_batch,                 # [batch_size,2]      tf placeholder
              batch_weights,               # [batch_size] now batch index % 4 - different sources, even - low variance, odd - high variance
              disp_diff_cap =         10.0, # cap disparity difference to this value (give up on large errors)
              disp_diff_slope=         0.0, #allow squared error to grow above disp_diff_cap
              absolute_disparity =     False, #when false there should be no activation on disparity output ! 
              use_confidence =         False, 
              lambda_conf_avg =        0.01,
              lambda_conf_pwr =        0.1,
              conf_pwr =               2.0,
              gt_conf_offset =         0.08,
              gt_conf_pwr =            1.0,
              error2_offset =          0.0025, # 0.0, # 0.0025, # (0.05^2) ~= coring
              disp_wmin =              1.0,    # minimal disparity to apply weight boosting for small disparities
              disp_wmax =              8.0,    # maximal disparity to apply weight boosting for small disparities
              use_out =                False):  # use calculated disparity for disparity weight boosting (False - use target disparity)
               
    with tf.name_scope("BatchLoss"):
        """
        Here confidence should be after relU. Disparity - may be also if absolute, but no activation if output is residual disparity
        """
        tf_lambda_conf_avg = tf.constant(lambda_conf_avg, dtype=tf.float32, name="tf_lambda_conf_avg")
        tf_lambda_conf_pwr = tf.constant(lambda_conf_pwr, dtype=tf.float32, name="tf_lambda_conf_pwr")
        tf_conf_pwr =        tf.constant(conf_pwr,        dtype=tf.float32, name="tf_conf_pwr")
        tf_gt_conf_offset =  tf.constant(gt_conf_offset,  dtype=tf.float32, name="tf_gt_conf_offset")
        tf_gt_conf_pwr =     tf.constant(gt_conf_pwr,     dtype=tf.float32, name="tf_gt_conf_pwr")
        tf_num_tiles =       tf.shape(gt_ds_batch)[0]
        tf_0f =              tf.constant(0.0,             dtype=tf.float32, name="tf_0f")
        tf_1f =              tf.constant(1.0,             dtype=tf.float32, name="tf_1f")
        tf_maxw =            tf.constant(1.0,             dtype=tf.float32, name="tf_maxw")
        tf_disp_diff_cap2=   tf.constant(disp_diff_cap*disp_diff_cap,  dtype=tf.float32, name="disp_diff_cap2")
        tf_disp_diff_slope=  tf.constant(disp_diff_slope, dtype=tf.float32, name="disp_diff_slope")
        
        if gt_conf_pwr == 0:
            w = tf.ones((out_batch.shape[0]), dtype=tf.float32,name="w_ones")
        else:
            w_slice = tf.reshape(gt_ds_batch[:,1],[-1],                     name = "w_gt_slice")
            
            w_sub =   tf.subtract      (w_slice, tf_gt_conf_offset,         name = "w_sub")
            w_clip =  tf.maximum(w_sub, tf_0f,                              name = "w_clip")
            if gt_conf_pwr == 1.0:
                w = w_clip
            else:
                w=tf.pow(w_clip, tf_gt_conf_pwr, name = "w_pow")
    
        if use_confidence:
            tf_num_tilesf =      tf.cast(tf_num_tiles, dtype=tf.float32,     name="tf_num_tilesf")
            conf_slice =     tf.reshape(out_batch[:,1],[-1],                 name = "conf_slice")
            conf_sum =       tf.reduce_sum(conf_slice,                       name = "conf_sum")
            conf_avg =       tf.divide(conf_sum, tf_num_tilesf,              name = "conf_avg")
            conf_avg1 =      tf.subtract(conf_avg, tf_1f,                    name = "conf_avg1")
            conf_avg2 =      tf.square(conf_avg1,                            name = "conf_avg2")
            cost2 =          tf.multiply (conf_avg2, tf_lambda_conf_avg,     name = "cost2")
    
            iconf_avg =      tf.divide(tf_1f, conf_avg,                      name = "iconf_avg")
            nconf =          tf.multiply (conf_slice, iconf_avg,             name = "nconf") #normalized confidence
            nconf_pwr =      tf.pow(nconf, conf_pwr,                         name = "nconf_pwr")
            nconf_pwr_sum =  tf.reduce_sum(nconf_pwr,                        name = "nconf_pwr_sum")
            nconf_pwr_offs = tf.subtract(nconf_pwr_sum, tf_1f,               name = "nconf_pwr_offs")
            cost3 =          tf.multiply (conf_avg2, nconf_pwr_offs,         name = "cost3")
            w_all =          tf.multiply (w, nconf,                          name = "w_all")
        else:
            w_all = w
#            cost2 = 0.0
#            cost3 = 0.0    
        # normalize weights
        w_sum =              tf.reduce_sum(w_all,                            name = "w_sum")
        iw_sum =             tf.divide(tf_1f, w_sum,                         name = "iw_sum")
        w_norm =             tf.multiply (w_all, iw_sum,                     name = "w_norm")
        
        disp_slice =         tf.reshape(out_batch[:,0],[-1],                 name = "disp_slice")
        d_gt_slice =         tf.reshape(gt_ds_batch[:,0],[-1],               name = "d_gt_slice")
        
        td_flat =        tf.reshape(target_disparity_batch,[-1],         name = "td_flat")
        if absolute_disparity:
            adisp =          disp_slice
        else:
            adisp =          tf.add(disp_slice, td_flat,                     name = "adisp")
        out_diff =           tf.subtract(adisp, d_gt_slice,                  name = "out_diff")
            
            
        out_diff2 =          tf.square(out_diff,                             name = "out_diff2")
        pre_cap0 =           tf.abs(out_diff,                                name = "pre_cap0")
        pre_cap =            tf.multiply(pre_cap0, tf_disp_diff_slope,       name = "pre_cap")
        diff_cap =           tf.add(pre_cap, tf_disp_diff_cap2,              name = "diff_cap")
        out_diff2_capped =   tf.minimum(out_diff2, diff_cap,                 name = "out_diff2_capped")
        out_wdiff2 =         tf.multiply (out_diff2_capped, w_norm,          name = "out_wdiff2")
        
        cost1 =              tf.reduce_sum(out_wdiff2,                       name = "cost1")
        
        out_diff2_offset =   tf.subtract(out_diff2, error2_offset,           name = "out_diff2_offset")
        out_diff2_biased =   tf.maximum(out_diff2_offset, 0.0,               name = "out_diff2_biased")
        
        # calculate disparity-based weight boost
        if use_out:
            dispw =          tf.clip_by_value(adisp, disp_wmin, disp_wmax,   name = "dispw")
        else:
            dispw =          tf.clip_by_value(td_flat, disp_wmin, disp_wmax, name = "dispw")
        dispw_boost =        tf.divide(disp_wmax, dispw,                     name = "dispw_boost")
        dispw_comp =         tf.multiply (dispw_boost, w_norm,               name = "dispw_comp") #HERE??

        if batch_weights.shape[0] > 1:
            dispw_batch =        tf.multiply (dispw_comp,  batch_weights,    name = "dispw_batch")# apply weights for high/low variance and sources
        else:
            dispw_batch =        tf.multiply (dispw_comp,  tf_1f,            name = "dispw_batch")# apply weights for high/low variance and sources


        dispw_sum =          tf.reduce_sum(dispw_batch,                      name = "dispw_sum")
        idispw_sum =         tf.divide(tf_1f, dispw_sum,                     name = "idispw_sum")
        dispw_norm =         tf.multiply (dispw_batch, idispw_sum,           name = "dispw_norm")
        
        out_diff2_wbiased =  tf.multiply(out_diff2_biased, dispw_norm,       name = "out_diff2_wbiased")
#        out_diff2_wbiased =  tf.multiply(out_diff2_biased, w_norm,       name = "out_diff2_wbiased")
        cost1b =             tf.reduce_sum(out_diff2_wbiased,                name = "cost1b")
        
        if use_confidence:
            cost12 =         tf.add(cost1b, cost2,                           name = "cost12")
            cost123 =        tf.add(cost12, cost3,                           name = "cost123")    
            
            return cost123, disp_slice, d_gt_slice, out_diff,out_diff2, w_norm, out_wdiff2, cost1
        else:
            return cost1b,  disp_slice, d_gt_slice, out_diff,out_diff2, w_norm, out_wdiff2, cost1
        
        
def weightsLoss(inp_weights,
                tile_layers,
                tile_side,
                wborders_zero):
                
                       # [batch_size,(1..2)] tf_result
#                weights_lambdas):  # single lambda or same length as inp_weights.shape[1]
    """
    Enforcing 'smooth' weights for the input 2d correlation tiles
    @return mean squared difference for each weight and average of 8 neighbors divided by mean squared weights
    """
    weight_ortho = 1.0
    weight_diag  = 0.7
    sw = 4.0 * (weight_ortho + weight_diag)
    weight_ortho /= sw
    weight_diag /=  sw
#    w_neib = tf.const([[weight_diag,  weight_ortho, weight_diag],
#                       [weight_ortho, -1.0,         weight_ortho],
#                       [weight_diag,  weight_ortho, weight_diag]])
    #WBORDERS_ZERO
    with tf.name_scope("WeightsLoss"):
        # Adding 1 tile border
#        tf_inp =     tf.reshape(inp_weights[:TILE_LAYERS * TILE_SIZE,:], [TILE_LAYERS, FILE_TILE_SIDE, FILE_TILE_SIDE, inp_weights.shape[1]], name = "tf_inp")
        tf_inp =     tf.reshape(inp_weights[:tile_layers * tile_side * tile_side,:], [tile_layers, tile_side, tile_side, inp_weights.shape[1]], name = "tf_inp")
        if wborders_zero:
            tf_zero_col = tf.constant(0.0, dtype=tf.float32, shape=[tf_inp.shape[0], tf_inp.shape[1], 1,                   tf_inp.shape[3]], name = "tf_zero_col")
            tf_zero_row = tf.constant(0.0, dtype=tf.float32, shape=[tf_inp.shape[0], 1 ,              tf_inp.shape[2] + 2, tf_inp.shape[3]], name = "tf_zero_row")
            tf_inp_ext_h = tf.concat([tf_zero_col,                 tf_inp,       tf_zero_col                 ], axis = 2, name ="tf_inp_ext_h")
            tf_inp_ext   = tf.concat([tf_zero_row,                 tf_inp_ext_h, tf_zero_row                 ], axis = 1, name ="tf_inp_ext")
        else:
            tf_inp_ext_h = tf.concat([tf_inp       [:, :,  :1, :], tf_inp,       tf_inp      [:,   :, -1:, :]], axis = 2, name ="tf_inp_ext_h")
            tf_inp_ext   = tf.concat([tf_inp_ext_h [:, :1, :,  :], tf_inp_ext_h, tf_inp_ext_h[:, -1:,   :, :]], axis = 1, name ="tf_inp_ext")
        
        s_ortho = tf_inp_ext[:,1:-1,:-2,:] + tf_inp_ext[:,1:-1, 2:,:] + tf_inp_ext[:,1:-1,:-2,:] + tf_inp_ext[:,1:-1, 2:, :] 
        s_corn =  tf_inp_ext[:, :-2,:-2,:] + tf_inp_ext[:, :-2, 2:,:] + tf_inp_ext[:,2:,  :-2,:] + tf_inp_ext[:,2:  , 2:, :]
        w_diff =  tf.subtract(tf_inp, s_ortho * weight_ortho + s_corn * weight_diag, name="w_diff") 
        w_diff2 = tf.multiply(w_diff, w_diff,                                        name="w_diff2") 
        w_var =   tf.reduce_mean(w_diff2,                                            name="w_var")
        w2_mean = tf.reduce_mean(inp_weights * inp_weights,                          name="w2_mean")
        w_rel =   tf.divide(w_var, w2_mean,                                          name= "w_rel")
        return w_rel # scalar, cost for weights non-smoothness in 2d
