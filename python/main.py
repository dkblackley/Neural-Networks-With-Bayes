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
EPOCHS = 3
DEBUG = True  # Toggle this to only run for 3% of the training data
ENABLE_GPU = False  # Toggle this to enable or disable GPU
BATCH_SIZE = 32
image_size = 224

if ENABLE_GPU:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

data_plot = data_plotting.DataPlotting()


composed = transforms.Compose([
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                # randomly crop out 10% of the total image
                                transforms.Resize((int((image_size/100) * 10) + image_size,
                                                   int((image_size/100) * 10) + image_size), Image.LANCZOS),
                                data_loading.RandomCrop(image_size),
                                transforms.ToTensor(),
                                # call helper.get_mean_and_std(data_set) to get mean and std
                                transforms.Normalize(mean=[0.3630, 0.0702, 0.0546], std=[0.3992, 0.3802, 0.4071])
                               ])

train_data = data_loading.data_set("Training_meta_data/ISIC_2019_Training_Metadata.csv", "ISIC_2019_Training_Input", labels_path="Training_meta_data/ISIC_2019_Training_GroundTruth.csv",  transforms=composed)
test_data = data_loading.data_set("Test_meta_data/ISIC_2019_Test_Metadata.csv", "ISIC_2019_Test_Input", transforms=composed)

"""weights = list(train_data.count_classes().values())
weights.pop()  # Remove the Unknown class
"""
weights = [4522, 12875, 3323, 867, 2624, 239, 253, 628]
# make a validation set
val_data, train_data = random_split(train_data, [2331, 23000])


train_set = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_set = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
test_set = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)


network = model.Classifier(image_size)
network.to(device)

optim = optimizer.Adam(network.parameters(), lr=0.001)

class_weights = torch.FloatTensor(weights).to(device)
loss_function = nn.CrossEntropyLoss(weight=class_weights)


def train(verbose=False):
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

        if (verbose):

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

        accuracy, val_loss, _ = test(val_set, verbose=verbose)
        val_losses.append(val_loss)
        val_accuracy.append(accuracy)

    return intervals, val_losses, train_losses, val_accuracy, train_accuracy


def test(testing_set, verbose=False):
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
                confusion_matrix[answer.item()][real_answer.item()] += 1

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

    if (verbose):

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

def predict(data_set, data_loader):
    """
    Predicts on a data set without labels
    :param data_set: data set to predict on
    :param data_loader: data loader to get the filename from
    :return: a list of lists holding the networks predictions for each class
    """
    batch = 0
    predictions = [['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']]

    for i_batch, sample_batch in enumerate(tqdm(data_set)):

        batch += 1


        image_batch = sample_batch['image']
        label_batch = sample_batch['label']

        image_batch = image_batch.to(device)

        outputs = network(image_batch, dropout=False)

        for i in range (0, BATCH_SIZE):
            try:
                answer = []
                answer = outputs[i].tolist()
                answer.insert(0, data_loader.get_filename(i + (BATCH_SIZE * i_batch)))
                predictions.append(answer)
            # for the last batch, which won't be perfectly of size BATCH_SIZE
            except Exception as e:
                break

    return predictions


def train_new_net():
    """
    Trains a network, saving the parameters and the losses/accuracies over time
    :return:
    """
    intervals, val_losses, train_losses, val_accuracies, train_accuracies = train(verbose=True)
    data_plot.plot_loss(intervals, val_losses, train_losses)
    data_plot.plot_validation(intervals, val_accuracies, train_accuracies)

    helper.save_net(network, "saved_model/model_parameters")
    helper.write_csv(val_losses, "saved_model/val_losses.csv")
    helper.write_csv(train_losses, "saved_model/train_losses.csv")
    helper.write_csv(val_accuracies, "saved_model/val_accuracies.csv")
    helper.write_csv(train_accuracies, "saved_model/train_accuracies.csv")

    _, __, confusion_matrix = test(val_set, verbose=True)
    data_plot.plot_confusion(confusion_matrix, "Validation Set")

    _, __, confusion_matrix = test(train_set, verbose=True)
    data_plot.plot_confusion(confusion_matrix, "Training Set")

#helper.plot_samples(train_data, data_plot)

train_new_net()


network = helper.load_net("saved_models/Classifier params 2 25 epochs/model_parameters")

predictions = predict(test_set, test_data)
helper.write_csv(predictions, "saved_model/predictions.csv")

_, __, confusion_matrix = test(val_set, verbose=True)

data_plot.plot_confusion(confusion_matrix, "Validation Set")

