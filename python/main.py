"""
Main.py: The file responsible for being the entrypoint into the program,
Deals with things like weight balancing, training and testing methods and
calling other classes for plotting results of the network
"""

import torch
import torch.optim as optimizer
from torchvision import transforms
from PIL import Image
from torch.utils.data import random_split, WeightedRandomSampler
import data_loading
import data_plotting
import helper
import model
import torch.nn as nn
from tqdm import tqdm

LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC', 8: 'UNK'}
EPOCHS = 25
DEBUG = False  # Toggle this to only run for 3% of the training data
ENABLE_GPU = False  # Toggle this to enable or disable GPU

if ENABLE_GPU:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

data_plot = data_plotting.DataPlotting()


composed = transforms.Compose([
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                transforms.Resize((256, 256), Image.LANCZOS),
                                data_loading.RandomCrop(224),
                                transforms.ToTensor(),
                                # call helper.get_mean_and_std(data_set) to get mean and std
                                transforms.Normalize(mean=[0.3630, 0.0702, 0.0546], std=[0.3992, 0.3802, 0.4071])
                               ])

train_data = data_loading.data_set("Training_meta_data/ISIC_2019_Training_Metadata.csv", "ISIC_2019_Training_Input", labels_path="Training_meta_data/ISIC_2019_Training_GroundTruth.csv",  transforms=composed)

weights = list(train_data.count_classes().values())
weights.pop()  # Remove the Unknown class

# make a validation set
val_data, train_data = random_split(train_data, [2331, 23000])


train_set = torch.utils.data.DataLoader(train_data, batch_size=30, shuffle=True)
val_set = torch.utils.data.DataLoader(val_data, batch_size=30, shuffle=True)


network = model.Classifier()
network.to(device)

optim = optimizer.Adam(network.parameters(), lr=0.001)


# Different methods for weight calculating, TODO: remove this at some point
"""
#weights = [(total / (4522)), (total / (12875)), (total / (3323)), (total / (867)), (total / (2624)), (total / (239)), (total / (253)), (total / (628))]
#weights = [((4522) / total), ((12875) / total), ((3323) / total), ((867) / total), ((2624) / total), ((239) / total), ((253) / total), ((628) / total), 0.0]
#weights = [(1 / (4522)), (1 / (12875)), (1 / (3323)), (1 / (867)), (1 / (2624)), (1 / (239)), (1 / (253)), (1 / (628))]
#weights = [(1 - (4522) / total), (1 - (12875) / total), (1 - (3323) / total), (1 - (867) / total), (1 - (2624) / total), (1 - (239) / total), (1 - (253) / total), (1 - (628) / total), 0.0]
#weights = [1/1002, 1/6034, 1/990, 1/295, 1/462, 1/104, 1/104, 1/128, 0]
#weights = np.multiply(6034, weights)
#weights = 1.0 / torch.Tensor([4522, 12875, 3323, 867, 2624, 239, 253, 628])
#weights = [4522, 12875, 3323, 867, 2624, 239, 253, 628]
#class_weights = [1 - (x / sum(weights)) for x in weights]
#class_weights = torch.tensor(np.multiply(6034, lossWeights), dtype = dtype)
#loss_function = nn.CrossEntropyLoss(weight=class_weights)
"""

class_weights = torch.FloatTensor(weights).to(device)
loss_function = nn.CrossEntropyLoss(weight=class_weights)


def train(verboose=False):
    """
    trains the network while also recording the accuracy of the network on the training data
    :param verboose: If true dumps out debug info about which classes the network is predicting when correct and incorrect
    :return: returns the number of epochs, a list of the validation set losses per epoch, a list of the
    training set losses per epoch, a list of the validation accuracy per epoch and the
    training accuracy per epoch
    """
    print("\nTraining Network")

    intervals = []
    val_losses = []
    train_losses = []
    val_accuracy = []
    train_accuracy = []

    for epoch in range(EPOCHS):

        losses = []

        correct = 0
        total = 0
        incorrect = 0
        correct_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0, 'UNK': 0}
        incorrect_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0, 'UNK': 0}

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

            percentage = (i_batch / len(train_set)) * 100  # Used for Debugging

            losses.append(loss.item())
            index = 0

            for output in outputs:
                answer = torch.argmax(output)
                real_answer = label_batch[index]
                index += 1

                if answer == real_answer:
                    label = LABELS[answer.item()]
                    correct_count[label] += 1
                    correct += 1
                else:
                    label = LABELS[answer.item()]
                    incorrect_count[label] += 1
                    incorrect += 1
                total += 1

            if percentage >= 3 and DEBUG:
                print(loss)
                break

        accuracy = (correct / total) * 100

        if (verboose):

            print("\n Correct Predictions: ")
            for label, count in correct_count.items():
                print(f"{label}: {count / correct * 100}%")

            print("\n Incorrect Predictions: ")
            for label, count in incorrect_count.items():
                print(f"{label}: {count / incorrect * 100}%")

            print(f"\nCorrect = {correct}")
            print(f"Total = {total}")
            print(f"Training Accuracy = {accuracy}%")

        intervals.append(epoch + 1)
        train_losses.append(sum(losses) / len(losses))
        print(f"Training loss: {sum(losses) / len(losses)}")

        train_accuracy.append(accuracy)

        accuracy, val_loss, _ = test(val_set, verboose=verboose)
        val_losses.append(val_loss)
        val_accuracy.append(accuracy)

    return intervals, val_losses, train_losses, val_accuracy, train_accuracy


def test(testing_set, verboose=False):
    """
    Use to test the network on a given data set
    :param testing_set: The data set that the network will predict for
    :param verboose: Dumps out debug data showing which class the network is predicting for
    :return: accuracy of network on dataset, loss of network and also a confusion matrix in the
    form of a list of lists
    """
    correct = 0
    total = 0
    incorrect = 0
    correct_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0, 'UNK': 0}
    incorrect_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0, 'UNK': 0}
    losses = []
    confusion_matrix = []

    for i in range(9):
        confusion_matrix.append([0, 0, 0, 0, 0, 0, 0, 0, 0])

    print("\nTesting Data...")

    with torch.no_grad():
        for i_batch, sample_batch in enumerate(tqdm(testing_set)):
            image_batch = sample_batch['image']
            label_batch = sample_batch['label']

            image_batch, label_batch = image_batch.to(device), label_batch.to(device)

            outputs = network(image_batch, dropout=False)
            loss = loss_function(outputs, label_batch)

            losses.append(loss.item())

            index = 0

            for output in outputs:

                answer = torch.argmax(output)
                real_answer = label_batch[index]
                confusion_matrix[real_answer.item()][answer.item()] += 1

                index += 1

                if answer == real_answer:
                    label = LABELS[answer.item()]
                    correct_count[label] += 1
                    correct += 1
                else:
                    label = LABELS[answer.item()]
                    incorrect_count[label] += 1
                    incorrect += 1
                total += 1

    average_loss = (sum(losses) / len(losses))
    accuracy = (correct / total) * 100

    if (verboose):

        print("\n Correct Predictions: ")
        for label, count in correct_count.items():
            print(f"{label}: {count / correct * 100}%")

        print("\n Incorrect Predictions: ")
        for label, count in incorrect_count.items():
            print(f"{label}: {count / incorrect * 100}%")

    print(f"\nCorrect = {correct}")
    print(f"Total = {total}")

    print(f"Test Accuracy = {accuracy}%")
    print(f"Test Loss = {average_loss}")

    return accuracy, average_loss, confusion_matrix


intervals, val_losses, train_losses, val_accuracies, train_accuracies = train(verboose=True)

data_plot.plot_loss(intervals, val_losses, train_losses)
data_plot.plot_validation(intervals, val_accuracies, train_accuracies)

helper.save_net(network, "Saved_model/model_parameters")
network = helper.load_net("Saved_model/model_parameters")

_, __, confusion_matrix = test(val_set, verboose=True)

data_plot.plot_confusion(confusion_matrix)

