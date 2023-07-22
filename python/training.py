import helper
import torch
import torch.optim as optimizer
from torchvision import transforms
from torch.utils.data import random_split, SubsetRandomSampler, Subset, WeightedRandomSampler
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import os

# Import other files
import data_loading
import data_plotting
import testing
import helper
import model

train_data = data_loading.data_set("ISIC_2019_Training_Input", labels_path="Training_meta_data/ISIC_2019_Training_GroundTruth.csv",  transforms=composed_train)
test_data = data_loading.data_set("ISIC_2019_Training_Input", labels_path="Training_meta_data/ISIC_2019_Training_GroundTruth.csv",  transforms=composed_test)
ISIC_data = data_loading.data_set("ISIC_2019_Test_Input",  transforms=composed_test)

def get_data_sets(plot=False):
    """
    Splits the data sets into train, test and validation sets
    :param plot: If true, plot some samples form each set
    :return: the DataLoader objects, ready to be called from for each set
    """

    # 70% split to train set, use 2/3rds of the remaining data (20%) for the testing set
    indices = list(range(len(train_data)))
    split_train = int(np.floor(0.7 * len(indices)))
    split_test = int(np.floor(0.66667 * (len(indices) - split_train)))

    temp_idx, train_idx = indices[split_train:], indices[:split_train]
    valid_idx, test_idx = temp_idx[split_test:], temp_idx[:split_test]

    weighted_train_idx = []

    for c in range(0, len(train_idx)):
        label = train_data.get_label(train_idx[c])
        weighted_idx = sampler_weights[label]
        weighted_train_idx.append(weighted_idx)

    weighted_train_sampler = WeightedRandomSampler(weights=weighted_train_idx,
                                                   num_samples=len(weighted_train_idx), replacement=True)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Don't shuffle the testing set for MC_DROPOUT
    testing_data = Subset(test_data, test_idx)

    training_set = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=weighted_train_sampler,
                                               shuffle=False)
    valid_set = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, sampler=valid_sampler)
    testing_set = torch.utils.data.DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=False)
    ISIC_set = torch.utils.data.DataLoader(ISIC_data, batch_size=BATCH_SIZE, shuffle=False)

    if plot:

        # Show some test images
        data_plot = data_plotting.DataPlotting(test_data, test_idx, 12, 14)
        helper.plot_set(training_set, data_plot, 0, 5)
        helper.plot_set(valid_set, data_plot, 0, 5)
        helper.plot_set(testing_set, data_plot, 0, 5)
        helper.plot_set(ISIC_set, data_plot, 0, 5)

    return training_set, valid_set, testing_set, ISIC_set, len(test_idx), len(train_idx), len(valid_idx), test_idx

train_set, val_set, test_set, ISIC_set, test_size, train_size, val_size, test_indexes = get_data_sets(plot=True)

data_plot = data_plotting.DataPlotting(test_data, test_indexes, 12, 14)
loss_function = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
val_loss_function = nn.CrossEntropyLoss(weight=val_weights, reduction='mean')
def train(root_dir, current_epoch, val_losses, train_losses, val_accuracy, train_accuracy, verbose=False):
    """
    Trains the network, saving the model with the best loss and the best accuracy as it goes.
    :param root_dir: Directory to save the model to
    :param current_epoch: The epoch to resume training from
    :param val_losses: The previous losses on the Validation set
    :param train_losses: The previous losses on the Training set
    :param val_accuracy: The previous Accuracies on the Validation set
    :param train_accuracy: The previous Accuracies on the training set
    :param verbose: if True, dumps out extra information regarding what the neural network has been predicting on
    :return: the train and val losses as well as the train and val accuracies
    """

    intervals = []

    # Set the best accuracy and loss values if none have been passed in.
    if not val_accuracy:
        best_val = 0
    else:
        best_val = max(val_accuracy)

    if not val_losses:
        best_loss = 1000000
    else:
        best_loss = min(val_losses)

    for i in range(0, len(val_losses)):
        intervals.append(i)

    print("\nTraining Network...")

    for epoch in range(current_epoch, EPOCHS + current_epoch):

        # Make sure network is in train mode
        network.train()

        losses = []
        correct = 0
        total = 0
        incorrect = 0
        correct_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}
        incorrect_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}

        print(f"\nEpoch {epoch + 1} of {EPOCHS + current_epoch}:")

        for i_batch, sample_batch in enumerate(tqdm(train_set)):
            image_batch = sample_batch['image'].to(device)
            label_batch = sample_batch['label'].to(device)

            outputs = network(image_batch, drop_samples=SAMPLES, dropout=True)
            loss = loss_function(outputs, label_batch)

            if BBB:
                loss += network.BBB_loss

            loss.backward()
            optim.step()
            optim.zero_grad()
            scheduler.step()
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
        train_accuracy.append(accuracy)
        print(f"Training loss: {sum(losses) / len(losses)}")

        accuracy, val_loss = test(val_set, verbose=verbose)
        val_losses.append(val_loss)
        val_accuracy.append(accuracy)

        if not os.path.isdir(root_dir):
            os.mkdir(root_dir)

        data_plot.plot_loss(root_dir, intervals, val_losses, train_losses)
        data_plot.plot_validation(root_dir, intervals, val_accuracy, train_accuracy)
        helper.save_network(network, optim, scheduler, val_losses, train_losses, val_accuracy, train_accuracy, root_dir)

        if best_val < max(val_accuracy):
            helper.save_network(network, optim, scheduler, val_losses, train_losses, val_accuracy, train_accuracy,
                                root_dir)
            best_val = max(val_accuracy)
            data_plot.plot_loss(root_dir, intervals, val_losses, train_losses)
            data_plot.plot_validation(root_dir, intervals, val_accuracy, train_accuracy)

        if best_loss > min(val_losses):
            helper.save_network(network, optim, scheduler, val_losses, train_losses, val_accuracy, train_accuracy,
                                root_dir)
            best_loss = min(val_losses)
            data_plot.plot_loss(root_dir, intervals, val_losses, train_losses)
            data_plot.plot_validation(root_dir, intervals, val_accuracy, train_accuracy)

    data_plot.plot_loss(root_dir, intervals, val_losses, train_losses)
    data_plot.plot_validation(root_dir, intervals, val_accuracy, train_accuracy)
    helper.save_network(network, optim, scheduler, val_losses, train_losses, val_accuracy, train_accuracy, root_dir)

    return intervals, val_losses, train_losses, val_accuracy, train_accuracy


def test(testing_set, verbose=False):
    """
    Used to test the network on the validation set
    :param testing_set: The set to be sample from for images and labels
    :param verbose: If True, prints out extra debug information about how the network is guessing
    :param dropout: If True, apply dropout at test time
    :return:
    """

    # Make sure network is in eval mode
    network.eval()

    correct = 0
    total = 0
    incorrect = 0
    correct_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}
    incorrect_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}
    losses = []

    print("\nTesting Data...")

    with torch.no_grad():
        for i_batch, sample_batch in enumerate(tqdm(testing_set)):
            image_batch = sample_batch['image'].to(device)
            label_batch = sample_batch['label'].to(device)

            outputs = network(image_batch, drop_samples=SAMPLES, sample=True, dropout=TRAIN_MC_DROPOUT)
            loss = val_loss_function(outputs, label_batch)

            if BBB:
                loss += network.BBB_loss

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

            if total >= BATCH_SIZE * 2 and DEBUG:
                break

    average_loss = (sum(losses) / len(losses))
    accuracy = (correct / total) * 100

    if (verbose):
        print("\n Correct Predictions: ")
        for label, count in correct_count.items():
            if correct == 0:
                print(f"{label}: {0}%")
            else:
                print(f"{label}: {count / correct * 100}%")

        print("\n Incorrect Predictions: ")
        for label, count in incorrect_count.items():
            print(f"{label}: {count / incorrect * 100}%")

    print(f"\nCorrect = {correct}")
    print(f"Total = {total}")

    print(f"Test Accuracy = {accuracy}%")
    print(f"Test Loss = {average_loss}")

    return accuracy, average_loss
