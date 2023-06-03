import numpy as np
import pandas as pd
import geotorch
import torch
import json

import os
import rasterio
import gdal

import matplotlib.image as image
import matplotlib.pyplot as plt
from PIL import Image

# ------------- PRINTING RBG portions of TIFF image, Exploratory Data Analysis ------------- #
FILEPATH = 'WYVERN_Data\WYVERN_AVIRISCALI_20181009t181942_v0_1_0.tiff'

# Converting tiff file into Numpy Array
file = rasterio.open(FILEPATH)
print(file.descriptions)
imageArr = torch.tensor(file.read()) ## Into array format, convert into tensor, easier format for later

# Printing dimensions, should have 32 matrices, one for each measured spectrum of light
print(imageArr.shape) 

# What does this mean?
band750 = 23
band705 = 19

# Cleaning outlier elements
imageArr[abs(imageArr) >= 9999] = np.nan
# RENDVI Calculation # Metric for checking health of plants, soil, etc
rendviArr = (imageArr[band750] - imageArr[band705]) / (imageArr[band750] + imageArr[band705])

# Plotting first image
# After averaging, RENDVI values should be in between -1, 1
#print(rendviArr)
#plt.imshow(rendviArr, vmin=-1, vmax=1)
#plt.show()

rows = 2
cols = 5

fig = plt.figure(figsize=(10,10))

for i in range(10, 20):
    
    fig.add_subplot(rows, cols, i-9)
    plt.imshow(imageArr[i])
    plt.title('Band #' + str(i+1))

plt.show()




# ------------- Experimenting with GeoJSON Files, format ------------- #
FILEPATH = "i15_crop_mapping_2019\\2019-Converted.geojson"

geojson2019 = json.load(open(FILEPATH))
print(geojson2019)

