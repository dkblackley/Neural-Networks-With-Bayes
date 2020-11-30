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

class dataSet(Dataset):

    def __init__(self, meta_path, labels_path=False, transforms=False):

        self.train_image_dir = "ISIC_2019_Training_Input/"
        self.metadata = pd.read_csv(meta_path)
        self.file_names = os.listdir(self.train_image_dir)
        self.file_names.sort()

        if labels_path:
            self.labels = pd.read_csv(labels_path)
            self.classes = self.labels.columns[1:10].values
        else:
            self.labels = False
        if transforms:
            self.transforms = transforms


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        file_name = str(index)
        file_name = self.file_names[index]

        full_path = os.path.join(self.train_image_dir, file_name)

        image = io.imread(full_path)
        label = self.get_class_name(self.labels.iloc[index].values)

        if self.transforms:
            image = self.transforms(image)

        return {'image': image, "label": label.item(0)}


    def get_class_name(self, numbers):
        index = np.where(numbers == 1.0)
        return (self.classes[index[0] - 1])

class reScale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, data):

        image, labels = data['image'], data['label']
        height, width = image.shape[:2]

        # If the user sent in a single int for size, i.e. a square image where width = height
        if isinstance(self.output_size, int):
            if height > width:
                height, width = self.output_size * height / width, self.output_size
            else:
                height, width = self.output_size, self.output_size * width / height
        else:
            height, width = self.output_size

        height, width = int(height), int(width)
        image = transform.resize(image, (height, width))

        return {'image': image, 'label': labels}


class randomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        image, label = data['image'], data['label']

        height, width = image.shape[:2]
        new_height, new_width = self.output_size

        top = np.random.randint(0, height - new_height)
        left = np.random.randint(0, width - new_width)

        image = image[top: top + new_height, left: left + new_width]

        return {'image': image, 'label': label}

class toTensor(object):

    def __call__(self, data):
        image, label = data['image'], data['label']

        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'label': label}

