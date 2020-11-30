from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm

class dataSet():

    def __init__(self, meta_path, labels_path=False):

        self.train_image_dir = "ISIC_2019_Training_Input/"
        self.metadata = pd.read_csv(meta_path)

        if labels_path:
            self.labels = pd.read_csv(labels_path)
            self.classes = self.labels.columns[1:10].values
        else:
            self.labels = False


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        file_name = str(index)
        file_name = "ISIC_" + file_name.zfill(7) + ".jpg"

        full_path = os.path.join(self.train_image_dir, file_name)

        image = io.imread(full_path)
        label = self.get_class_name(self.labels.iloc[index].values)

        return {'image': image, "label": label}


    def get_class_name(self, numbers):
        index = np.where(numbers == 1.0)
        return (self.classes[index[0] - 1])

