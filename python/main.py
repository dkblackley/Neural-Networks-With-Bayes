import torch
import torch.optim as optimizer
from torchvision import transforms
from PIL import Image
import dataLoading
import dataPlotting
import model
import torch.nn as nn
from tqdm import tqdm

LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC', 8: 'UNK'}
EPOCHS = 1

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
train_set = torch.utils.data.DataLoader(train_data, batch_size=30, shuffle=True, num_workers=0)


network = model.Classifier()

optim = optimizer.Adam(network.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

print("Training Network")

for epoch in range(EPOCHS):
    print(f"Epoch {epoch} of {EPOCHS}:")
    for i_batch, sample_batch in enumerate(tqdm(train_set)):
        image_batch = sample_batch['image']
        label_batch = sample_batch['label']

        optim.zero_grad()
        outputs = network(image_batch)
        loss = loss_function(outputs, label_batch)
        loss.backward()
        optim.step()

print(loss)














"""
for i in range(len(train_data)):
    data = train_data[i]
    dataPlot.show_data(data)
    print(i, data['image'].size(), LABELS[data['label']])
    if i == 3:
        break



for i_batch, sample_batch in enumerate(train_set):
    print(i_batch, sample_batch['image'].size(),
          sample_batch['label'].size())

    if i_batch == 3:
        dataPlot.show_batch(sample_batch, 3)
        break



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


