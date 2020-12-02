import torch
import torchvision
import torch.optim as optimizer
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from efficientnet_pytorch import EfficientNet

import dataLoading
import dataPlotting

classes = ['MEL' 'NV' 'BCC' 'AK' 'BKL' 'DF' 'VASC' 'SCC' 'UNK']

dataPlot = dataPlotting.dataPlotting()


composed = transforms.Compose([
                                #dataLoading.randomRotation([0, 90, 180, 270]),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                transforms.Resize((256, 256), Image.LANCZOS),
                                dataLoading.randomCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) #TODO calculate mean and std
                               ])

train_data = dataLoading.dataSet("Training_meta_data/ISIC_2019_Training_Metadata.csv", "Training_meta_data/ISIC_2019_Training_GroundTruth.csv", transforms=composed)
train_set = torch.utils.data.DataLoader(train_data, batch_size=9, shuffle=True, num_workers=0)

for i in range(len(train_data)):
    data = train_data[i]
    dataPlot.show_data(data)
    print(i, data['image'].size(), data['label'])
    if i == 3:
        break

for i_batch, sample_batch in enumerate(train_set):
    print(i_batch, sample_batch['image'].size(),
          sample_batch['label'].size())

    if i_batch == 3:
        dataPlot.show_data(sample_batch[0])
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


