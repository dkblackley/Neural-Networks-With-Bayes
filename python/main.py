"""
Main.py: The file responsible for being the entrypoint into the program,
Deals with things like weight balancing, training and testing methods and
calling other classes for plotting results of the network
"""

import torch
import torch.optim as optimizer
from torchvision import transforms
from torch.utils.data import random_split, SubsetRandomSampler, SequentialSampler
import numpy as np
import data_loading
import data_plotting
import helper
import model
import torch.nn as nn
from tqdm import tqdm

LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC', 8: 'UNK'}
EPOCHS = 3
DEBUG = False  # Toggle this to only run for 1% of the training data
ENABLE_GPU = False  # Toggle this to enable or disable GPU
BATCH_SIZE = 32
SOFTMAX = True
MC_DROPOUT = False
FORWARD_PASSES = 100
BBB = False
image_size = 224

if ENABLE_GPU:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

data_plot = data_plotting.DataPlotting()

# One percent of the overall image size
image_percent = image_size/100
image_five_percent = int(image_percent * 5)

composed_train = transforms.Compose([
                                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                                transforms.ColorJitter(brightness=0.2),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                # Skew the image by 1% of its total size
                                transforms.RandomAffine(0, shear=0.01),
                                transforms.ToTensor(),
                                transforms.RandomErasing(p=0.2, scale=(0.001, 0.005)),
                                transforms.RandomErasing(p=0.2, scale=(0.001, 0.005)),
                                transforms.RandomErasing(p=0.2, scale=(0.001, 0.005)),
                                transforms.RandomErasing(p=0.2, scale=(0.001, 0.005)),
                                #transforms.RandomErasing(p=0.25, scale=(image_percent/image_size/10, image_percent/image_size/5)),
                                # call helper.get_mean_and_std(data_set) to get mean and std
                                transforms.Normalize(mean=[0.6685, 0.5296, 0.5244], std=[0.2247, 0.2043, 0.2158])
                               ])

"""composed_train = transforms.Compose([
                                transforms.Resize((image_size, image_size), Image.LANCZOS),
                                transforms.ToTensor()
                               ])"""

"""composed_test = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ColorJitter(brightness=0.2),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomAffine(0, shear=image_percent/image_size),
                                transforms.ToTensor(),
                                # call helper.get_mean_and_std(data_set) to get mean and std
                                transforms.Normalize(mean=[0.6786, 0.5344, 0.5273], std=[0.2062, 0.1935, 0.2064])
                               ])"""

train_data = data_loading.data_set("Training_meta_data/ISIC_2019_Training_Metadata.csv", "ISIC_2019_Training_Input", labels_path="Training_meta_data/ISIC_2019_Training_GroundTruth.csv",  transforms=composed_train)
#test_data = data_loading.data_set("Test_meta_data/ISIC_2019_Test_Metadata.csv", "ISIC_2019_Test_Input", transforms=composed_test)



"""weights = list(train_data.count_classes().values())
weights.pop()  # Remove the Unknown class"""



def get_data_sets(plot=False):

    indices = list(range(len(train_data)))
    split_train = int(np.floor(0.7 * len(train_data)))
    split_val = int(np.floor(0.33 * split_train))

    np.random.seed(1337)
    np.random.shuffle(indices)

    temp_idx, train_idx = indices[split_train:], indices[:split_train]
    valid_idx, test_idx = temp_idx[split_val:], temp_idx[:split_val]

    if (bool(set(test_idx) & set(valid_idx))):
        print("HERE")
    if (bool(set(test_idx) & set(train_idx))):
        print("HERE")
    if (bool(set(train_idx) & set(valid_idx))):
        print("HERE")

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # Don't shuffle the testing set for MC_DROPOUT
    test_sampler = SequentialSampler(test_idx)

    training_set = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)
    valid_set = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=valid_sampler)
    testing_set = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=test_sampler)

    if plot:
        helper.plot_set(training_set, data_plot, 0, 5)
        helper.plot_set(valid_set, data_plot, 0, 5)
        helper.plot_set(testing_set, data_plot, 0, 5)

    return training_set, valid_set, testing_set


train_set, val_set, test_set = get_data_sets(plot=True)

network = model.Classifier(image_size, dropout=0.5)
network.to(device)

optim = optimizer.Adam(network.parameters(), lr=0.001)

#weights = list(helper.count_classes(train_set, BATCH_SIZE).values())
weights = [3188, 8985, 2319, 602, 1862, 164, 170, 441]

new_weights = []
index = 0
for weight in weights:
    new_weights.append(sum(weights)/(8 * weight))
    index = index + 1

#weights = [4522, 12875, 3323, 867, 2624, 239, 253, 628]

class_weights = torch.Tensor(new_weights).to(device)
loss_function = nn.CrossEntropyLoss(weight=class_weights)


def train(current_epoch, val_losses, train_losses, val_accuracy, train_accuracy, verbose=False):
    """
    trains the network while also recording the accuracy of the network on the training data
    :param verboose: If true dumps out debug info about which classes the network is predicting when correct and incorrect
    :return: returns the number of epochs, a list of the validation set losses per epoch, a list of the
    training set losses per epoch, a list of the validation accuracy per epoch and the
    training accuracy per epoch
    """

    intervals = []

    for i in range(0, len(val_losses)):
        intervals.append(i)

    print("\nTraining Network")

    for epoch in range(current_epoch, EPOCHS + current_epoch):

        # Make sure network is in train mode
        network.train()

        losses = []
        correct = 0
        total = 0
        incorrect = 0
        correct_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0, 'UNK': 0}
        incorrect_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0, 'UNK': 0}

        print(f"\nEpoch {epoch + 1} of {EPOCHS + current_epoch}:")

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

            if percentage >= 1 and DEBUG:
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

        data_plot.plot_loss(intervals, val_losses, train_losses)
        data_plot.plot_validation(intervals, val_accuracy, train_accuracy)

        save_network(optim, val_losses, train_losses, val_accuracy, train_accuracy)

    return intervals, val_losses, train_losses, val_accuracy, train_accuracy


def test(testing_set, verbose=False):
    """
    Use to test the network on a given data set
    :param testing_set: The data set that the network will predict for
    :param verboose: Dumps out debug data showing which class the network is predicting for
    :return: accuracy of network on dataset, loss of network and also a confusion matrix in the
    form of a list of lists
    """

    # Make sure network is in eval mode
    network.eval()

    correct = 0
    total = 0
    incorrect = 0
    correct_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0, 'UNK': 0}
    incorrect_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0, 'UNK': 0}
    losses = []
    confusion_matrix = []

    for i in range(8):
        confusion_matrix.append([0, 0, 0, 0, 0, 0, 0, 0])

    print("\nTesting Data...")

    with torch.no_grad():
        for i_batch, sample_batch in enumerate(tqdm(testing_set)):
            image_batch = sample_batch['image']
            label_batch = sample_batch['label']

            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            with torch.no_grad():
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

            if total >= BATCH_SIZE * 2 and DEBUG:
                break

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

def save_network(optim, val_losses, train_losses, val_accuracies, train_accuracies):
    helper.save_net(network, optim, "saved_model/model_parameters")
    helper.write_csv(val_losses, "saved_model/val_losses.csv")
    helper.write_csv(train_losses, "saved_model/train_losses.csv")
    helper.write_csv(val_accuracies, "saved_model/val_accuracies.csv")
    helper.write_csv(train_accuracies, "saved_model/train_accuracies.csv")


def load_net(root_dir):

    val_losses = helper.read_csv(root_dir + "val_losses.csv")
    train_losses = helper.read_csv(root_dir + "train_losses.csv")
    val_accuracies = helper.read_csv(root_dir + "val_accuracies.csv")
    train_accuracies = helper.read_csv(root_dir + "train_accuracies.csv")
    network, optim = helper.load_net(root_dir + "model_parameters", image_size)

    return network, optim, len(train_losses), val_losses, train_losses, val_accuracies, train_accuracies

def confusion_array(arrays):

    new_arrays = []

    for array in arrays:
        new_array = []

        for item in array:
            new_array.append(round(item / (sum(array) + 1), 4))

        new_arrays.append(new_array)

    return new_arrays

def train_net(starting_epoch=0, val_losses=[], train_losses=[], val_accuracies=[], train_accuracies=[]):
    """
    Trains a network, saving the parameters and the losses/accuracies over time
    :return:
    """
    starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = train(
        starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies, verbose=True)

    if not DEBUG:

        _, __, confusion_matrix = test(val_set, verbose=True)
        data_plot.plot_confusion(confusion_matrix, "Validation Set")
        confusion_matrix = confusion_array(confusion_matrix)
        data_plot.plot_confusion(confusion_matrix, "Validation Set Normalized")

        _, __, confusion_matrix = test(train_set, verbose=True)
        data_plot.plot_confusion(confusion_matrix, "Training Set")
        confusion_matrix = confusion_array(confusion_matrix)
        data_plot.plot_confusion(confusion_matrix, "Training Set Normalized")

        _, __, confusion_matrix = test(test_set, verbose=True)
        data_plot.plot_confusion(confusion_matrix, "Testing Set")
        confusion_matrix = confusion_array(confusion_matrix)
        data_plot.plot_confusion(confusion_matrix, "Testing Set Normalized")

    return val_losses, train_losses, val_accuracies, train_accuracies

train_net()
#helper.plot_samples(train_data, data_plot)

network, optim, starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = load_net("saved_model/")

#EPOCHS = 10

#test(val_set, verbose=True)

train_net(starting_epoch=starting_epoch,
          val_losses=val_losses,
          train_losses=train_losses,
          val_accuracies=val_accuracies,
          train_accuracies=train_accuracies)


#predictions = predict(test_set)





