# Description
Class imagej_tiff to read multilayer tiff files and parse tags
* layers are stacked along depth (think RGB)
* parse imagej generated tags (50838 and 50839)

# Dependencies
* Python 3.5.2 (not strict)
* Pillow 5.1.0+ (strict)
* Numpy 1.14.2 (not strict)
* Matplotlib 2.2.2 (not strict)

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
