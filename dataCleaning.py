import numpy as np
import pandas as pd
import geopandas as gpd
import torchgeo
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import sklearn

import os
import rasterio
import gdal

import matplotlib.image as image
import matplotlib.pyplot as plt

# ------------- Importing TIFF data, Labbel Data, array conversion for manipulation, Cleaning ------------- #
FILEPATH = 'WYVERN_Data'
FILEPATH2018 = 'i15_crop_mapping_2018_shp\2018_Converted.geojson'
FILEPATH2019 = 'i15_crop_mapping_2019\2019-Converted.geojson'

# Will store WYVERN image data as array of torch.tensor types
X = pd.DataFrame()
y = pd.DataFrame()

'Looping through WYVERN Map data for TIFF files, GeoJSON'
for filename in os.listdir(FILEPATH):

    if filename.endswith('.tiff') or filename.endswith('.tif'):

        # ----------------------------------------#
        # Converting tiff file into Numpy Array
        file = rasterio.open(os.path.join(FILEPATH, filename))
        imageArray[abs(imageArray) >= 9999] = np.nan
        imageArray = torch.tensor(file.read())
        df2 = pd.DataFrame(imageArray)
        
        X.append(df2)

        # ----------------------------------------#
        # Extracting labels from geoJSON depending on year of filename.tiff
        # Pushing into y dataframe
        if filename.__contains__('2018'):
            
            # Must choose 2018 labels
            
        else:



# ------------- Creating DATASET class for PyTorch uses ------------- #
class tiffDataset(Dataset):
    image_size = 100
    def __init__(self, X, y):
        'Initialization. X => Training Images'
        self.X=X
        self.y=y

    def __len__(self):
        'Denotes total # of samples'
        return len(self.X)
    
    def __getitem__(self, index):
        'Generates item from sample by index'
        image = self.X[index]
        X = self.transform(image)
        return X
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(image_size),
        T.ToTensor()
    ])


# ------------- Splitting Data into Train, Test ------------- #
XTrain, XTest, yTrain, yTest = sklearn.model_selection.train_test_split(X, y, test_size = 0.25)