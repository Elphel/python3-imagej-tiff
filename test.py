#!/usr/bin/env python3
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import imagej_tiff as ijt

tiff = ijt.imagej_tiff('test.tiff')
print(tiff.nimages)
print(tiff.labels)
print(tiff.infos)
tiff.show_images(['X-corr','Y-corr',0,2])
plt.show()

