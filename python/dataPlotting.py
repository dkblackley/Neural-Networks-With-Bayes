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

class dataPlotting():

    def show_data(self, data):

        image = data['image']
        if torch.is_tensor(image):
            trsfm = transforms.ToPILImage(mode='RGB')
            image = trsfm(image)
            #image = image.numpy()
            #image = image.transpose((2, 1, 0))

        plt.figure()
        plt.axis('off')
        plt.imshow(image)
        plt.title(data['label'].item(0) + " Sample")
        plt.show()

# Helper function to show a batch
def show_batch(sample_batched):
    images_batch, landmarks_batch = \
        sample_batched['image'], sample_batched['label']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)


    plt.axis('off')
    plt.ioff()
    plt.show()