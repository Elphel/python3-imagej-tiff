#!/usr/bin/env python3

import time
import numpy as np

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


# MAIN
print("Importing TensorCrawl")
print_time()

import tensorflow as tf
import tensorflow.contrib.slim as slim

print("TensorCrawl imported")
print_time()

sess = tf.Session()

# (10,)
a_in = [0,1,2,3,4,5,6,7,8,9]
# (1,)
b_in = [1.5]
# (3,2)
c_in = [
  [0.1,0.2],
  [0.3,0.4],
  [0.5,0.6]
]

#print(np.array(a_in).shape)

a = tf.placeholder(tf.float32,[None,10])
b = tf.placeholder(tf.float32,[None,1])

c = tf.concat([a,b],axis=1)

#d = tf.constant(-1.0)
e = tf.maximum(a-1.0,0.0)

f = tf.placeholder(tf.float32,[None,3,2])

g = tf.maximum(f[:,:,1]+0.01,0)

res = sess.run(g,feed_dict={a:[a_in],b:[b_in],f:[c_in]})
#res = sess.run(a,feed_dict={a:a_in})

print(res.shape)
print(res)

print_time()
print(bcolors.OKGREEN+"time: "+str(time.time())+bcolors.ENDC)