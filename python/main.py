import torch
import torchvision
import torch.optim as optimizer
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

import dataLoading
import dataPlotting


dataPlot = dataPlotting.dataPlotting()


composed = transforms.Compose([
                                #dataLoading.randomRotation([0, 90, 180, 270]),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                transforms.Resize((1024, 1024), Image.LANCZOS),
                                dataLoading.randomCrop(800),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) #TODO calculate mean and std
                               ])

train_data = dataLoading.dataSet("Training_meta_data/ISIC_2019_Training_Metadata.csv", "Training_meta_data/ISIC_2019_Training_GroundTruth.csv", transforms=composed)


for i in range(len(train_data)):
    data = train_data[i]
    dataPlot.show_data(data)

    if i == 3:
        break








"""

dataPlot = dataPlotting.dataPlotting()
scale = dataLoading.reScale(1024)
crop = dataLoading.randomCrop((700, 1000))
composed = transforms.Compose([dataLoading.reScale(1024),
                               dataLoading.randomCrop(800)])

train_data = dataLoading.dataSet("Training_meta_data/ISIC_2019_Training_Metadata.csv", "Training_meta_data/ISIC_2019_Training_GroundTruth.csv")

item = train_data.__getitem__(24)

# Apply each of the above transforms on sample.
fig = plt.figure()

for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(item)
    dataPlot.show_data(transformed_sample)

plt.show()


dataPlot.show_data(item)

"""


