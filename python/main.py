"""
Main.py: The file responsible for being the entrypoint into the program,
Deals with things like weight balancing, training and testing methods and
calling other classes for plotting results of the network
"""

import torch
import sys
import torch.optim as optimizer
from torchvision import transforms
from PIL import Image
from torch.utils.data import random_split, WeightedRandomSampler, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import data_loading
import data_plotting
import helper
import model
import torch.nn as nn
from tqdm import tqdm

LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC', 8: 'UNK'}
EPOCHS = 0
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

composed_train = transforms.Compose([
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                # randomly crop out 10% of the total image
                                transforms.Resize((int((image_size/100) * 10) + image_size,
                                                   int((image_size/100) * 10) + image_size), Image.LANCZOS),
                                data_loading.RandomCrop(image_size),
                                transforms.ToTensor(),
                                # call helper.get_mean_and_std(data_set) to get mean and std
                                transforms.Normalize(mean=[0.6786, 0.5344, 0.5273], std=[0.2062, 0.1935, 0.2064])
                               ])

composed_test = transforms.Compose([
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                # randomly crop out 10% of the total image
                                transforms.Resize((int((image_size/100) * 10) + image_size,
                                                   int((image_size/100) * 10) + image_size), Image.LANCZOS),
                                data_loading.RandomCrop(image_size),
                                transforms.ToTensor(),
                                # call helper.get_mean_and_std(data_set) to get mean and std
                                transforms.Normalize(mean=[0.6284, 0.5216, 0.5166], std=[0.2341, 0.2143, 0.2244])
                               ])

train_data = data_loading.data_set("Training_meta_data/ISIC_2019_Training_Metadata.csv", "ISIC_2019_Training_Input", labels_path="Training_meta_data/ISIC_2019_Training_GroundTruth.csv",  transforms=composed_train)
test_data = data_loading.data_set("Test_meta_data/ISIC_2019_Test_Metadata.csv", "ISIC_2019_Test_Input", transforms=composed_test)



"""weights = list(train_data.count_classes().values())
weights.pop()  # Remove the Unknown class"""



def get_data_sets(plot=False):

    indices = list(range(len(train_data)))
    split = int(np.floor(0.85 * len(train_data)))
    np.random.seed(1337)
    np.random.shuffle(indices)
    valid_idx, train_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    training_set = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)
    valid_set = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=valid_sampler)
    # Don't shuffle the testing set for MC_DROPOUT
    testing_set = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    if plot:
        helper.plot_set(training_set, data_plot, 0, 5)
        helper.plot_set(valid_set, data_plot, 0, 5)
        helper.plot_set(testing_set, data_plot, 0, 5)

    return training_set, valid_set, testing_set


train_set, val_set, test_set = get_data_sets(plot=True)

network = model.Classifier(image_size, dropout=0.5)
network.to(device)

optim = optimizer.Adam(network.parameters(), lr=0.001)

weights = [4522, 12875, 3323, 867, 2624, 239, 253, 628]

class_weights = torch.FloatTensor(weights).to(device)
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

        save_network(val_losses, train_losses, val_accuracy, train_accuracy)

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
            with torch.no_grad():
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

def softmax_pred(outputs, data_loader, i_batch):

    predictions = []

    for i in range(0, BATCH_SIZE):

        try:
            answers = outputs[i].tolist()

            new_list = []
            for answer in answers:
                new_list.append(answer)

            if SOFTMAX:
              new_list.append(1.0 - max(new_list))

            for c in range(0, len(new_list)):
                new_list[c] = '{:.17f}'.format(new_list[c])

            answers = new_list

            answers.insert(0, data_loader.get_filename(i + (BATCH_SIZE * i_batch))[:-4])
            predictions.append(answers)
        # for the last batch, which won't be perfectly of size BATCH_SIZE
        except Exception as e:
            break

    return predictions

def monte_carlo(data_set, data_loader, forward_passes):
    n_classes = 8
    n_samples = len(data_loader)
    soft_max = nn.Softmax(dim=1)
    drop_predictions = np.empty((0, n_samples, n_classes))

    for i in tqdm(range(forward_passes)):

        print(f"\n\n Forward pass {i + 1} of {forward_passes}\n")
        predictions = np.empty((0, n_classes))

        with tqdm(total=len(data_set), position=0, leave=True) as pbar:
            for i_batch, sample_batch in enumerate(tqdm((data_set), position=0, leave=True)):
                image_batch = sample_batch['image']

                with torch.no_grad():
                    outputs = soft_max(network(image_batch, dropout=True))

                for output in outputs:
                    predictions = np.vstack((predictions, output.cpu().numpy()))


        drop_predictions = np.vstack((drop_predictions, predictions[np.newaxis, :, :]))

    mean = np.mean(drop_predictions, axis=0)  # shape (n_samples, n_classes)
    variance = np.var(drop_predictions, axis=0) # shape (n_samples, n_classes)

    mean = mean.tolist()
    variance = variance.tolist()

    print("\nAttaching Filenames")
    i = 0
    for preds in mean:

        preds.insert(0, data_loader.get_filename(i)[:-4])
        preds.append(sum(variance[i]))

        for c in range(1, len(preds)):
            preds[c] = '{:.17f}'.format(preds[c])

        i = i + 1

    return mean


def predict(data_set, data_loader):
    """
    Predicts on a data set without labels
    :param data_set: data set to predict on
    :param data_loader: data loader to get the filename from
    :return: a list of lists holding the networks predictions for each class
    """

    print("\nPredicting on Test set")

    batch = 0

    if SOFTMAX:
        predictions = [['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']]
        for i_batch, sample_batch in enumerate(tqdm(data_set)):

            batch += 1

            image_batch = sample_batch['image']

            image_batch = image_batch.to(device)
            soft_max = nn.Softmax(dim=1)
            with torch.no_grad():
                outputs = soft_max(network(image_batch, dropout=False))
            predictions.extend(softmax_pred(outputs, data_loader, i_batch))

    elif MC_DROPOUT:
        predictions = monte_carlo(data_set, data_loader, FORWARD_PASSES)
        predictions.insert(0, ['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'])


    return predictions


def save_network(val_losses, train_losses, val_accuracies, train_accuracies):
    helper.save_net(network, "saved_model/model_parameters")
    helper.write_csv(val_losses, "saved_model/val_losses.csv")
    helper.write_csv(train_losses, "saved_model/train_losses.csv")
    helper.write_csv(val_accuracies, "saved_model/val_accuracies.csv")
    helper.write_csv(train_accuracies, "saved_model/train_accuracies.csv")


def load_net(root_dir):

    val_losses = helper.read_csv(root_dir + "val_losses.csv")
    train_losses = helper.read_csv(root_dir + "train_losses.csv")
    val_accuracies = helper.read_csv(root_dir + "val_accuracies.csv")
    train_accuracies = helper.read_csv(root_dir + "train_accuracies.csv")
    network = helper.load_net(root_dir + "model_parameters", image_size)

    return network, len(train_losses), val_losses, train_losses, val_accuracies, train_accuracies

def confusion_array(arrays):

    new_arrays = []

    for array in arrays:
        new_array = []

        for item in array:
            new_array.append(int((item / (sum(array) + 1)) * 100))

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
        data_plot.plot_confusion(confusion_matrix, "Validation Set (%)")

        _, __, confusion_matrix = test(train_set, verbose=True)
        data_plot.plot_confusion(confusion_matrix, "Training Set")
        confusion_matrix = confusion_array(confusion_matrix)
        data_plot.plot_confusion(confusion_matrix, "Training Set (%)")

    return val_losses, train_losses, val_accuracies, train_accuracies

#train_net()
#helper.plot_samples(train_data, data_plot)

network, starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = load_net("saved_model/")

#test(val_set, verbose=True)

"""train_net(starting_epoch=starting_epoch,
          val_losses=val_losses,
          train_losses=train_losses,
          val_accuracies=val_accuracies,
          train_accuracies=train_accuracies)"""


predictions = predict(test_set, test_data)

if SOFTMAX:
    helper.write_rows(predictions, "saved_model/softmax_predictions.csv")
elif MC_DROPOUT:
    helper.write_rows(predictions, "saved_model/dropout_predictions.csv")



