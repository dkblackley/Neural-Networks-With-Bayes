import torch
import torch.optim as optimizer
from torchvision import transforms
from PIL import Image
from torch.utils.data import random_split, WeightedRandomSampler
import dataLoading
import dataPlotting
import model
import torch.nn as nn
import numpy as np
from tqdm import tqdm

LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC', 8: 'UNK'}
EPOCHS = 3
DEBUG = False
ENABLE_GPU = True

if ENABLE_GPU:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

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

#train_data.count_classes()
train_data.make_equal()
train_data.count_classes()

# make an improptu test set
test_data, train_data = random_split(train_data, [244, 8800])


train_set = torch.utils.data.DataLoader(train_data, batch_size=30, shuffle=True)




network = model.Classifier()
network.to(device)

optim = optimizer.Adam(network.parameters(), lr=0.001)

total = len(train_data) - 1331
#weights = [(total / (4522)), (total / (12875)), (total / (3323)), (total / (867)), (total / (2624)), (total / (239)), (total / (253)), (total / (628))]
#weights = [((4522) / total), ((12875) / total), ((3323) / total), ((867) / total), ((2624) / total), ((239) / total), ((253) / total), ((628) / total), 0.0]
#weights = [(1 / (4522)), (1 / (12875)), (1 / (3323)), (1 / (867)), (1 / (2624)), (1 / (239)), (1 / (253)), (1 / (628))]
#weights = [(1 - (4522) / total), (1 - (12875) / total), (1 - (3323) / total), (1 - (867) / total), (1 - (2624) / total), (1 - (239) / total), (1 - (253) / total), (1 - (628) / total), 0.0]
#weights = [1/1002, 1/6034, 1/990, 1/295, 1/462, 1/104, 1/104, 1/128, 0]
#weights = np.multiply(6034, weights)
#weights = 1.0 / torch.Tensor([4522, 12875, 3323, 867, 2624, 239, 253, 628])
weights = [4522, 12875, 3323, 867, 2624, 239, 253, 628]

#class_weights = [1 - (x / sum(weights)) for x in weights]
class_weights = torch.FloatTensor(weights).to(device)
#class_weights = torch.tensor(np.multiply(6034, lossWeights), dtype = dtype)
#loss_function = nn.CrossEntropyLoss(weight=class_weights)
loss_function = nn.CrossEntropyLoss()


def plot_samples():

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



def train():
    print("Training Network")

    average_losses = []
    intervals = []

    for epoch in range(EPOCHS):
        interval = 10
        losses = []
        print(f"\nEpoch {epoch + 1} of {EPOCHS}:")
        for i_batch, sample_batch in enumerate(tqdm(train_set)):
            image_batch = sample_batch['image']
            label_batch = sample_batch['label']

            image_batch, label_batch = image_batch.to(device), label_batch.to(device)

            optim.zero_grad()
            outputs = network(image_batch, dropout=True)
            loss = loss_function(outputs, label_batch)
            loss.backward()
            optim.step()

            percentage = (i_batch/len(train_set)) * 100

            losses.append(loss.item())

            if percentage >= 30 and DEBUG:
                print(loss)
                break
            """if percentage >= interval:
                interval += 10
                print(loss)
                losses.append(loss.item())
                intervals.append((interval / 100) + epoch)"""
        average_losses = (sum(losses) / i_batch)
        print(f"loss: {loss}")
        if DEBUG:
            return losses, intervals
    return average_losses, intervals



def test():

    correct = 0
    total = 0
    incorrect = 0
    correct_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0, 'UNK': 0}
    incorrect_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0, 'UNK': 0}

    print("\nTesting Data...")
    with torch.no_grad():
        for i in tqdm(range(len(test_data))):
            data = test_data[i]
            image = data['image']
            real_label = data['label']

            output = torch.argmax(network(image[None, ...]))

            if output == real_label:
                label = LABELS[output.item()]
                correct_count[label] += 1
                correct += 1
            else:
                label = LABELS[output.item()]
                incorrect_count[label] += 1
                incorrect += 1
            total += 1

    print(f"\nCorrect = {correct}")
    print(f"Total = {total}")
    print(f"Accuracy = {(correct / total) * 100}%")

    print("\n Correct Predictions: ")
    for label, count in correct_count.items():
        print(f"{label}: {count / correct * 100}%")

    print("\n Incorrect Predictions: ")
    for label, count in incorrect_count.items():
        print(f"{label}: {count / incorrect * 100}%")



losses, intervals = train()
dataPlot.plot_loss(losses, intervals)
test()


