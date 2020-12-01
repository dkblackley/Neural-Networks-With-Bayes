from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class dataSet(Dataset):

    def __init__(self, meta_path, labels_path=False, transforms=None):

        self.train_image_dir = "ISIC_2019_Training_Input/"
        self.metadata = pd.read_csv(meta_path)
        self.file_names = os.listdir(self.train_image_dir)
        self.file_names.sort()
        self.transforms = transforms

        if labels_path:
            self.labels = pd.read_csv(labels_path)
            self.classes = self.labels.columns[1:10].values
        else:
            self.labels = False

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        file_name = self.file_names[index]

        full_path = os.path.join(self.train_image_dir, file_name)
        image = Image.open(full_path)
        label = self.get_class_name(self.labels.iloc[index].values)

        if self.transforms:
            image = self.transforms(image)

        data = {'image': image, "label": label}

        return data

    def get_class_name(self, numbers):
        index = np.where(numbers == 1.0)
        return (self.classes[index[0] - 1])

    def add_transforms(self, transforms):
        self.transforms = transforms



class randomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        image = np.asarray(image)

        height, width = image.shape[:2]
        new_height, new_width = self.output_size

        top = np.random.randint(0, height - new_height)
        left = np.random.randint(0, width - new_width)

        image = image[top: top + new_height, left: left + new_width]

        image = Image.fromarray(image, 'RGB')

        return image



class randomRotation(object):
    def __init__(self, angle):
        self.angles = angle

    def __call__(self, image):

        image = TF.rotate(image, np.random.choice(self.angles))

        return image
"""
class randomFlips(object):

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.randint(1, 100) > 50:
            image = TF.hflip(image)
        if random.randint(1, 100) > 50:
            image = TF.vflip(image)
        return {'image': image, 'label': label}

# TODO Add in calculating mean and std
class normalize(object):
    def __init__(self, dataset, mean=False, std=False):

        if mean and std:
            self.mean = mean
            self.std = std

    def __call__(self, data):
        image, label = data['image'], data['label']
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return {'image': image, 'label': label}
"""