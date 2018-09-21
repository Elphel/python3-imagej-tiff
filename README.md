# Description
Class imagej_tiff to read multilayer tiff files and parse tags
* layers are stacked along depth (think RGB)
* parse imagej generated tags (50838 and 50839)

# More info
* Presentation for CVPR2018: [Elphel_TP-CNN_slides.pdf](https://community.elphel.com/files/presentations/Elphel_TP-CNN_slides.pdf)
* [TIFF Image stacks for Machine Learning](https://wiki.elphel.com/wiki/Tiff_file_format_for_pre-processed_quad-stereo_sets#TIFF_image_stacks_for_ML)

# Samples
* [models/all/state_street/1527256815_150165/v01/ml/](https://community.elphel.com/3d+biquad/models/all/state_street/1527256815_150165/v01/ml/)
or
* go to [3d+biquad](https://community.elphel.com/3d+biquad/), open individual models and hit the light green button to ‘Download source files for ml’

# Dependencies
* Python 3.5.2 (not strict)
* Pillow 5.1.0+ (strict)
* Numpy 1.14.2 (not strict)
* Matplotlib 2.2.2 (not strict)

# Infer and convert model for use from Java
```
~$ python3 infer_qcds_01.py qcstereo_conf.xml data_sets
```
where: 
* **qcstereo_conf.xml** - config file
* **data_sets** - root dir for trained models/data/checkpoints

# Examples
```
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
```
