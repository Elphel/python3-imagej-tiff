# Description
Class imagej_tiff to read multilayer tiff files and parse tags
* layers are stacked along depth (think RGB)
* parse imagej generated tags (50838 and 50839)

# Examples
```
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import imagej_tiff as ijt

tiff = ijt.imagej_tiff('test.tiff')
print(ijt.labels)
print(ijt.infos)
print(ijt.infos)
tiff.show_images(['X-corr','Y-corr',0,2])
plt.show()
```
