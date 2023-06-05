# ------------- Python Imports
import numpy as np ## I wonder what this is
import pandas as pd ## I wonder what this is
import os ## Navigate Python OS
from PIL import Image ## Python lib to work with Image importing, processing
import rasterio

# ------------- PyTorch imports
import torch ## PyTorch
from torch.utils.data import DataLoader, Dataset ## In-house classes build to hold/iterate through data, including our image data
import torchvision ## PyTorch library containing tools specifically for Computer Vision
import torchvision.transforms as transforms ## Holds several common image transformation processing tools, to help clean data b4 training

## LOAD DATA

# ------------- STEP 1: Importing TIFF data, array conversion for manipulation ------------- #
FOLDERPATH = 'WYVERN_AVIRIS_CALIFORNIA_IMAGERY_2018-2019_SOURCE'
TENSORFOLDERPATH = 'WYVERN_DAT_TENSOR_FORMAT'
FILEPATH2018 = 'i15_crop_mapping_2018_shp\2018_Converted.geojson'
FILEPATH2019 = 'i15_crop_mapping_2019\2019-Converted.geojson'

## Setting up for GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Loading in appropriate data
images = []

toTensor = transforms.ToTensor() # Class which instantiates a function to convert images to normalized tensors

## Must iterate through FOLDERPATH folder to load in each .TIFF file with it's 23 layers seperately
for fileName in os.listdir(FOLDERPATH):
    if fileName.endswith('.tiff'):
        filePath = os.path.join(FOLDERPATH, fileName)
        print(f'file {fileName} opening.')
        image = toTensor(rasterio.open(filePath).read())
        print(f'file is now Tensor.')
        images.append(image)
        print('save in progress...')
        torch.save(image, f'image_{fileName}.pt')
        print('done saving')

print('Done')


