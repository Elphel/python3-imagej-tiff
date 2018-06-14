#!/usr/bin/env python3

__copyright__ = "Copyright 2018, Elphel, Inc."
__license__   = "GPL-3.0+"
__email__     = "oleg@elphel.com"

'''
Test:
  nvidia graphic card
  cuda installation
  tensorflow

Comment:
  With nvidia + tensorflow - any software update casually breaks everything

'''

import subprocess
import re

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



# STEP 1: print nvidia model
print(bcolors.BOLDWHITE+"NVIDIA devices:"+bcolors.ENDC)

p = subprocess.run("lspci | grep NVIDIA",shell=True,stdout=subprocess.PIPE)
out = p.stdout.strip().decode()
if len(out)==0:
  print(bcolors.FAIL+" not found (try 'lspci')"+bcolors.ENDC)
else:
  print(out)

# STEP 2: nvidia driver version
print(bcolors.BOLDWHITE+"NVIDIA driver version:"+bcolors.ENDC)
p = subprocess.run("cat /proc/driver/nvidia/version",shell=True,stdout=subprocess.PIPE)
out = p.stdout.strip().decode()
print(out)

# STEP 3: nvidia-smi - also some information about the graphics card and the driver
print(bcolors.BOLDWHITE+"Some more info from 'nvidia-smi':"+bcolors.ENDC)
p = subprocess.run("nvidia-smi",shell=True,stdout=subprocess.PIPE)
out = p.stdout.strip().decode()
print(out)


print(bcolors.OKGREEN+"END"+bcolors.ENDC)