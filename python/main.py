"""
Main.py: The file responsible for being the entrypoint into the program,
Deals with things like weight balancing, training and testing methods and
calling other classes for plotting results of the network
"""

EPOCHS = 100
UNKNOWN_CLASS = False
DEBUG = True #Toggle this to only run for 1% of the training data
ENABLE_GPU = False  # Toggle this to enable or disable GPU
BATCH_SIZE = 32
SOFTMAX = True
MC_DROPOUT = False
TRAIN_MC_DROPOUT = True
COST_MATRIX = False
TEST_COST_MATRIX = False
FORWARD_PASSES = 100
BBB = False
SAVE_DIR = "saved_model"

if TRAIN_MC_DROPOUT and BBB:
    exit()

import torch
import torch.optim as optimizer
from torchvision import transforms
from torch.utils.data import random_split, SubsetRandomSampler, Subset, WeightedRandomSampler, SequentialSampler
import numpy as np
import data_loading
import data_plotting
import testing
import helper
import model
import torch.nn as nn
from tqdm import tqdm
import os

LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC', 8: 'UNK'}
image_size = 224
test_indexes = []
test_size = 0
val_size = 0
train_size = 0
best_val = 0
best_loss = 0
np.random.seed(1337)

if ENABLE_GPU:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


#weights = [3620, 10297, 2661, 688, 2109, 187, 200, 502] #80%
weights = [3188, 8985, 2319, 602, 1862, 164, 170, 441] #70%


new_weights = []
sampler_weights = []
k = 0
q = 1
for weight in weights:
    new_weights.append(((sum(weights))/weight)**k)
    sampler_weights.append(((sum(weights))/weight)**q)

class_weights = torch.Tensor(new_weights)
sampler_weights = torch.Tensor(sampler_weights)

class_weights = class_weights.to(device)
sampler_weights = sampler_weights.to(device)
print(class_weights)
print(sampler_weights)

composed_train = transforms.Compose([
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(35),
                                transforms.Resize(int(image_size*1.5)),
                                transforms.CenterCrop(int(image_size*1.25)),
                                transforms.RandomAffine(0, shear=5),
                                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
                                transforms.ToTensor(),
                                transforms.RandomErasing(p=0.2, scale=(0.001, 0.005)),
                                transforms.RandomErasing(p=0.2, scale=(0.001, 0.005)),
                                transforms.RandomErasing(p=0.2, scale=(0.001, 0.005)),
                                transforms.RandomErasing(p=0.2, scale=(0.001, 0.005)),
                                # call helper.get_mean_and_std(data_set) to get mean and std
                                transforms.Normalize(mean=[0.6685, 0.5296, 0.5244], std=[0.2247, 0.2043, 0.2158])
                               ])

composed_test = transforms.Compose([
                                transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                # call helper.get_mean_and_std(data_set) to get mean and std
                                transforms.Normalize(mean=[0.6685, 0.5296, 0.5244], std=[0.2247, 0.2043, 0.2158])
                               ])

train_data = data_loading.data_set("Training_meta_data/ISIC_2019_Training_Metadata.csv", "ISIC_2019_Training_Input", labels_path="Training_meta_data/ISIC_2019_Training_GroundTruth.csv",  transforms=composed_train)
test_data = data_loading.data_set("Training_meta_data/ISIC_2019_Training_Metadata.csv", "ISIC_2019_Training_Input", labels_path="Training_meta_data/ISIC_2019_Training_GroundTruth.csv",  transforms=composed_test)

def get_data_sets(plot=False):


    if UNKNOWN_CLASS:
        scc_idx = []
        print("\nRemoving SCC class from train and validation sets")
        for i in tqdm(range(0, len(train_data))):
            if train_data[i]['label'] == 7:
                scc_idx.append(i)



        indices = list(range(len(train_data)))
        indices = [x for x in indices if x not in scc_idx]

        split_train = int(np.floor(0.7 * len(indices)))
        split_test = int(np.floor(0.6667 * (len(indices) - split_train)))

        temp_idx, train_idx = indices[split_train:], indices[:split_train]
        valid_idx, test_idx = temp_idx[split_test:], temp_idx[:split_test]

        for i in scc_idx:
            if i not in test_idx:
                test_idx.append(i)

    else:
        indices = list(range(len(train_data)))
        split_train = int(np.floor(0.7 * len(indices)))
        split_test = int(np.floor(0.6667 * (len(indices) - split_train)))

        temp_idx, train_idx = indices[split_train:], indices[:split_train]
        valid_idx, test_idx = temp_idx[split_test:], temp_idx[:split_test]

    weighted_train_idx = []
    for c in range(0, len(train_idx)):
        label = train_data.get_label(train_idx[c])
        weighted_idx = sampler_weights[label]
        #weighted_idx = 1
        weighted_train_idx.append(weighted_idx)

    weighted_train_sampler = WeightedRandomSampler(weights=weighted_train_idx, num_samples=len(weighted_train_idx), replacement=True)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Don't shuffle the testing set for MC_DROPOUT
    testing_data = Subset(test_data, test_idx)

    training_set = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=weighted_train_sampler, shuffle=False)
    valid_set = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=valid_sampler)
    testing_set = torch.utils.data.DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=False)

    if plot:
        data_plot = data_plotting.DataPlotting(UNKNOWN_CLASS, test_data, test_idx)

        helper.plot_set(training_set, data_plot, 0, 5)
        helper.plot_set(valid_set, data_plot, 0, 5)
        helper.plot_set(testing_set, data_plot, 0, 5)



    return training_set, valid_set, testing_set, len(test_idx), len(train_idx), len(valid_idx), test_idx

train_set, val_set, test_set, test_size, train_size, val_size, test_indexes = get_data_sets(plot=False)

data_plot = data_plotting.DataPlotting(UNKNOWN_CLASS, test_data, test_indexes)

#helper.count_classes(train_set, BATCH_SIZE)
#helper.count_classes(val_set, BATCH_SIZE)
#helper.count_classes(test_set, BATCH_SIZE)

"""train_names = []
val_names = []
test_names = []


for i_batch, sample_batch in enumerate(tqdm(train_set)):
    for name in sample_batch['filename']:
        train_names.append(name)

for i_batch, sample_batch in enumerate(tqdm(val_set)):
    for name in sample_batch['filename']:
        val_names.append(name)

for i_batch, sample_batch in enumerate(tqdm(test_set)):
    for name in sample_batch['filename']:
        test_names.append(name)

if bool(set(train_names) & set(val_names)):
    intersection = set(train_names).intersection(val_names)
    print(intersection)

if bool(set(test_names) & set(val_names)):
    intersection = set(test_names).intersection(val_names)
    print(intersection)

if bool(set(test_names) & set(train_names)):
    intersection = set(test_names).intersection(train_names)
    print(intersection)"""


"""summed = sum(weights)
new_weights = 164 / torch.Tensor(weights)

#new_weights = [weight/sum(weights) for weight in weights]
#print(new_weights)

#new_weights = [1.0 / weight for weight in weights]
#new_weights = new_weights / sum(new_weights)
print(new_weights)

new_weights = []
index = 0
for weight in weights:
    if UNKNOWN_CLASS:
        new_weights.append(sum(weights) / (7 * weight))
    else:
        new_weights.append(sum(weights)/(8 * weight))
    index = index + 1

new_weights = torch.Tensor(new_weights)
print(new_weights)"""
#weights = [4522, 12875, 3323, 867, 2624, 239, 253, 628]

#new_weights = helper.apply_cost_matrix()

if COST_MATRIX:
    loss_function = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
else:
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    val_loss_fuinction = nn.CrossEntropyLoss(weight=sampler_weights)

"""
if UNKNOWN_CLASS:
    network = model.Classifier(image_size, 7, class_weights, device, dropout=0.5, BBB=BBB)
    network.to(device)
    weights = [3188, 8985, 2319, 602, 1862, 164, 170]
else:
    network = model.Classifier(image_size, 8, class_weights, device, dropout=0.5, BBB=BBB)
    network.to(device)
    weights = [3188, 8985, 2319, 602, 1862, 164, 170, 441]

weights = [3620, 10297, 2661, 688, 2109, 187, 200, 502]
    
optim = optimizer.Adam(network.parameters(), lr=0.001, weight_decay=0.001)"""

#weights = list(helper.count_classes(train_set, BATCH_SIZE).values())


"""output_hook = model.OutputHook()
network.relu.register_forward_hook(output_hook)
activation_penalty = 0.00001"""


def train(root_dir, current_epoch, val_losses, train_losses, val_accuracy, train_accuracy, mc_val_losses, mc_val_accuracies, verbose=False):
    """
    trains the network while also recording the accuracy of the network on the training data
    :param verboose: If true dumps out debug info about which classes the network is predicting when correct and incorrect
    :return: returns the number of epochs, a list of the validation set losses per epoch, a list of the
    training set losses per epoch, a list of the validation accuracy per epoch and the
    training accuracy per epoch
    """

    #input("\nHave you remembered to move the model out of saved model before loading in the best model?")

    intervals = []
    avg_BBB_losses = []
    BBB_val_losses = []
    best_BBB_loss = 1000000
    if not val_accuracy:
        best_val = 0
        best_mc_val = 0
    else:
        best_val = max(val_accuracy)
        best_mc_val = max(val_accuracy)

    if not val_losses:
        best_loss = 1000000
        best_mc_loss = 1000000
    else:
        best_loss = min(val_losses)
        best_mc_loss = min(val_losses)

    for i in range(0, len(val_losses)):
        intervals.append(i)

    print("\nTraining Network")

    for epoch in range(current_epoch, EPOCHS + current_epoch):

        # Make sure network is in train mode
        network.train()

        losses = []
        BBB_losses = []
        correct = 0
        total = 0
        incorrect = 0
        correct_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}
        incorrect_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}

        print(f"\nEpoch {epoch + 1} of {EPOCHS + current_epoch}:")

        for i_batch, sample_batch in enumerate(tqdm(train_set)):
            image_batch = sample_batch['image'].to(device)
            label_batch = sample_batch['label'].to(device)

            if BBB:
                outputs = network(image_batch, labels=label_batch)
                efficientNet_loss = loss_function(outputs, label_batch)
                loss = network.BBB_loss + efficientNet_loss

            else:
                outputs = network(image_batch, dropout=True)
                loss = loss_function(outputs, label_batch)
            
            """activation_cost = 0
            for output in output_hook:
                activation_cost += torch.norm(output, 1)
            activation_cost *= activation_penalty
            loss += activation_cost"""
            
            """L1_reg = torch.tensor(0., requires_grad=True)
            for name, param in network.named_parameters():
                if 'weight' in name:
                    L1_reg = L1_reg + torch.norm(param, 1)

                loss = loss + 0.0001 * L1_reg"""
            
            loss.backward()
            optim.step()
            optim.zero_grad()
            scheduler.step()
            percentage = (i_batch / len(train_set)) * 100  # Used for Debugging
            
            if BBB:
                BBB_losses.append(loss.item())
                loss = efficientNet_loss

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

            """if percentage % 10 >= 9.85 and verbose:
                print("\n")
                print(loss)"""

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
        if BBB:
            avg_BBB_losses.append(sum(BBB_losses) / len(BBB_losses))
        print(f"Training loss: {sum(losses) / len(losses)}")

        train_accuracy.append(accuracy)

        accuracy, val_loss, BBB_val, _ = test(val_set, verbose=verbose)
        val_losses.append(val_loss)
        BBB_val_losses.append(BBB_val)
        val_accuracy.append(accuracy)

        if TRAIN_MC_DROPOUT:
            accuracy, val_loss, BBB_val, _ = test(val_set, verbose=verbose, drop_samples=10, dropout=True)
            mc_val_losses.append(val_loss)
            mc_val_accuracies.append(accuracy)
        
        if not os.path.isdir(root_dir):
            os.mkdir(root_dir)

        data_plot.plot_loss(root_dir, intervals, val_losses, train_losses)
        data_plot.plot_validation(root_dir, intervals, val_accuracy, train_accuracy)
        save_network(optim, scheduler, val_losses, train_losses, val_accuracy, train_accuracy, root_dir)

        if best_val < max(val_accuracy):
            save_network(optim, scheduler, val_losses, train_losses, val_accuracy, train_accuracy, root_dir + "best_model/")
            best_val = max(val_accuracy)
            data_plot.plot_loss(root_dir + "best_model/", intervals, val_losses, train_losses)
            data_plot.plot_validation(root_dir + "best_model/", intervals, val_accuracy, train_accuracy)

        if best_loss > min(val_losses):
            save_network(optim, scheduler, val_losses, train_losses, val_accuracy, train_accuracy, root_dir + "best_loss/")
            best_loss = min(val_losses)
            data_plot.plot_loss(root_dir + "best_loss/", intervals, val_losses, train_losses)
            data_plot.plot_validation(root_dir + "best_loss/", intervals, val_accuracy, train_accuracy)

        if TRAIN_MC_DROPOUT and best_mc_val < max(mc_val_accuracies):
            save_network(optim, scheduler, mc_val_losses, train_losses, mc_val_accuracies, train_accuracy, root_dir + "best_mc_model/")
            best_mc_val = max(mc_val_accuracies)
            data_plot.plot_loss(root_dir + "best_mc_model/", intervals, mc_val_losses, train_losses)
            data_plot.plot_validation(root_dir + "best_mc_model/", intervals, mc_val_accuracies, train_accuracy)

        if TRAIN_MC_DROPOUT and best_mc_loss > min(mc_val_losses):
            save_network(optim, scheduler, mc_val_losses, train_losses, mc_val_accuracies, train_accuracy, root_dir + "best_mc_loss/")
            best_mc_loss = min(mc_val_losses)
            data_plot.plot_loss(root_dir + "best_mc_loss/", intervals, mc_val_losses, train_losses)
            data_plot.plot_validation(root_dir + "best_mc_loss/", intervals, mc_val_accuracies, train_accuracy)

        if BBB:
            temp = [e for e in range(0, len(BBB_val_losses))]
            data_plot.plot_loss(root_dir + "BBB_", temp, BBB_val_losses, avg_BBB_losses)
            helper.write_csv(avg_BBB_losses, root_dir + "BBB_train_losses.csv")
            helper.write_csv(BBB_val_losses, root_dir + "BBB_val_losses.csv")
            
            if best_BBB_loss > min(BBB_val_losses):
                save_network(optim, scheduler, BBB_val_losses, avg_BBB_losses, val_accuracy, train_accuracy, root_dir + "best_BBB_loss/")
                best_BBB_loss = min(BBB_val_losses)
                data_plot.plot_loss(root_dir + "best_BBB_loss/", temp, BBB_val_losses, avg_BBB_losses)
                data_plot.plot_validation(root_dir + "best_BBB_loss/", temp, val_accuracy, train_accuracy)
    data_plot.plot_loss(root_dir, intervals, val_losses, train_losses)
    data_plot.plot_validation(root_dir, intervals, val_accuracy, train_accuracy)
    save_network(optim, scheduler, val_losses, train_losses, val_accuracy, train_accuracy, root_dir)

    return intervals, val_losses, train_losses, val_accuracy, train_accuracy


def test(testing_set, drop_samples=1, verbose=False, dropout=False):
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
    correct_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}
    incorrect_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}
    losses = []
    BBB_losses = []
    confusion_matrix = []

    for i in range(8):
        confusion_matrix.append([0, 0, 0, 0, 0, 0, 0, 0])

    print("\nTesting Data...")

    with torch.no_grad():
        for i_batch, sample_batch in enumerate(tqdm(testing_set)):
            image_batch = sample_batch['image'].to(device)
            label_batch = sample_batch['label'].to(device)

            with torch.no_grad():

                if BBB:
                    outputs = network(image_batch, labels=label_batch, sample=True, dropout=dropout)
                    efficientNet_loss = val_loss_fuinction(outputs, label_batch)
                    loss = network.BBB_loss + efficientNet_loss

                else:
                    outputs = network(image_batch, drop_samples=drop_samples, dropout=dropout)
                    loss = val_loss_fuinction(outputs, label_batch)
                    
            if BBB:
                BBB_losses.append(loss.item())
                loss = efficientNet_loss

            losses.append(loss.item())
            index = 0
            for output in outputs:

                answer = torch.argmax(output)
                real_answer = label_batch[index]
                confusion_matrix[answer.item()][real_answer.item()] += 1
                #confusion_matrix[real_answer.item()][answer.item()] += 1

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
    average_BBB = 0
    if BBB:
        average_BBB = (sum(BBB_losses) / len(BBB_losses))
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

    return accuracy, average_loss, average_BBB, confusion_matrix

def save_network(optim, scheduler, val_losses, train_losses, val_accuracies, train_accuracies, root_dir):
    
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)

    helper.save_net(network, optim, scheduler, root_dir + "model_parameters")
    helper.write_csv(val_losses, root_dir + "val_losses.csv")
    helper.write_csv(train_losses, root_dir + "train_losses.csv")
    helper.write_csv(val_accuracies, root_dir + "val_accuracies.csv")
    helper.write_csv(train_accuracies, root_dir + "train_accuracies.csv")


def load_net(root_dir, output_size):

    val_losses = helper.read_csv(root_dir + "val_losses.csv")
    train_losses = helper.read_csv(root_dir + "train_losses.csv")
    val_accuracies = helper.read_csv(root_dir + "val_accuracies.csv")
    train_accuracies = helper.read_csv(root_dir + "train_accuracies.csv")
    network, optim, scheduler = helper.load_net(root_dir + "model_parameters", image_size, output_size, device, class_weights)

    network = network.to(device)

    return network, optim, scheduler, len(train_losses), val_losses, train_losses, val_accuracies, train_accuracies


def train_net(root_dir, starting_epoch=0, val_losses=[], train_losses=[], val_accuracies=[], train_accuracies=[], mc_val_accuracies=[], mc_val_losses=[]):
    """
    Trains a network, saving the parameters and the losses/accuracies over time
    :return:
    """
    starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = train(root_dir, starting_epoch,
                                                                                       val_losses, train_losses,
                                                                                       val_accuracies, train_accuracies,
                                                                                       mc_val_accuracies, mc_val_losses,
                                                                                       verbose=True)

    return val_losses, train_losses, val_accuracies, train_accuracies


def print_metrics(root_dir):
    helper.remove_last_row(costs_sr)
    helper.remove_last_row(costs_mc)
    helper.remove_last_row(costs_BBB)

    data_plot.plot_risk_coverage([predictions_mc, predictions_softmax, predictions_BBB], root_dir, "Risk Coverage")

    data_plot.plot_true_cost_coverage_by_class([costs_mc, costs_sr, costs_BBB], root_dir,
                                               "Average Test cost by Classes using LEC")
    data_plot.plot_true_cost_coverage([costs_mc, costs_sr], root_dir, "Coverage by Average Test cost", uncertainty=False)

    data_plot.plot_true_cost_coverage_by_class([predictions_mc, predictions_softmax], root_dir,
                                               "Average Test cost by Classes using raw Probabilities with Flattened Matrix",
                                               costs=False, flatten=True)

    data_plot.plot_true_cost_coverage_by_class([costs_mc, costs_sr], root_dir,
                                      "Average Test cost by Classes using LEC")

    data_plot.plot_true_cost_coverage_by_class([predictions_mc, predictions_softmax], root_dir,
                                               "Average Test cost by Classes using raw Probabilities", costs=False)



    data_plot.plot_true_cost_coverage([predictions_mc, predictions_softmax], root_dir,
                                      "Average Test cost using Raw Probabilities with Flattened Matrix", costs=False, flatten=True, uncertainty=True)
    data_plot.plot_true_cost_coverage([predictions_mc, predictions_softmax], root_dir,
                                      "Average Test cost using Raw Probabilities", costs=False)
    data_plot.plot_true_cost_coverage([costs_mc, costs_sr], root_dir, "Average Test cost using LEC")

    data_plot.plot_cost_coverage([costs_mc, costs_sr], root_dir, "Coverage by Lowest Expected cost")

    data_plot.count_sampels_in_intervals(predictions_mc, root_dir, "Number of Samples in each Interval MC Dropout", 5)
    data_plot.count_sampels_in_intervals(predictions_mc, root_dir, "Number of Samples in each Interval MC Dropout (Without probabilites below 0.2)", 5, skip_first=True)
    data_plot.count_sampels_in_intervals(predictions_softmax, root_dir, "Number of Samples in each Interval Softmax", 5)
    data_plot.count_sampels_in_intervals(predictions_softmax, root_dir, "Number of Samples in each Interval Softmax (Without probabilites below 0.2)", 5, skip_first=True)


    data_plot.plot_calibration(predictions_mc, "MC Dropout Reliability Diagram", root_dir, 5)
    data_plot.plot_calibration(predictions_softmax, "Softmax Reliability Diagram", root_dir, 5)

    costs_with_entropy_mc = helper.attach_last_row(costs_mc, predictions_mc)
    costs_with_entropy_sr = helper.attach_last_row(costs_sr, predictions_softmax)
    data_plot.plot_cost_coverage([costs_with_entropy_mc, costs_with_entropy_sr], root_dir, "Risk Coverage by Uncertainty", uncertainty=False)

    correct_mc, incorrect_mc, uncertain_mc = helper.get_correct_incorrect(predictions_mc, test_data, test_indexes, TEST_COST_MATRIX)
    print(f"MC Accuracy: {len(correct_mc)/(len(correct_mc) + len(incorrect_mc)) * 100}")

    correct_sr, incorrect_sr, uncertain_sr = helper.get_correct_incorrect(predictions_softmax, test_data, test_indexes, TEST_COST_MATRIX)
    print(f"SM Accuracy: {len(correct_sr) / (len(correct_sr) + len(incorrect_sr)) * 100}")

    data_plot.plot_correct_incorrect_uncertainties(correct_mc, incorrect_mc, root_dir, "MC Dropout Variance by class", by_class=True, prediction_index=0)
    data_plot.plot_correct_incorrect_uncertainties(correct_sr, incorrect_sr, root_dir, "Softmax Response Variance by class", by_class=True, prediction_index=0)

    data_plot.plot_correct_incorrect_uncertainties(correct_mc, incorrect_mc, root_dir, "MC Dropout Variance across predictions")
    data_plot.plot_correct_incorrect_uncertainties(correct_sr, incorrect_sr, root_dir, "Softmax Response Entropies across predictions")

    data_plot.average_uncertainty_by_class(correct_mc, incorrect_mc, root_dir, "MC Dropout Accuracies by Prediction")
    data_plot.average_uncertainty_by_class(correct_sr, incorrect_sr, root_dir, "Softmax Response Accuracies by Prediction")

    data_plot.plot_risk_coverage([predictions_mc, predictions_softmax], root_dir, "Risk Coverage")

    confusion_matrix = helper.make_confusion_matrix(predictions_softmax, test_data, test_indexes, True)
    data_plot.plot_confusion(confusion_matrix, root_dir, "Softmax Response on Test Set with Cost matrix")
    confusion_matrix = helper.confusion_array(confusion_matrix, dimension=1)
    data_plot.plot_confusion(confusion_matrix, root_dir, "Softmax Response Test Set Normalized with Cost matrix")

    confusion_matrix = helper.make_confusion_matrix(predictions_mc, test_data, test_indexes, True)
    data_plot.plot_confusion(confusion_matrix, root_dir, "MC dropout Test Set with Cost matrix")
    confusion_matrix = helper.confusion_array(confusion_matrix, dimension=1)
    data_plot.plot_confusion(confusion_matrix, root_dir, "MC dropout Test Set Normalized with Cost matrix")

    confusion_matrix = helper.make_confusion_matrix(predictions_softmax, test_data, test_indexes, False)
    data_plot.plot_confusion(confusion_matrix, root_dir, "Softmax Response on Test Set without Cost matrix")
    confusion_matrix = helper.confusion_array(confusion_matrix, dimension=1)
    data_plot.plot_confusion(confusion_matrix, root_dir, "Softmax Response Test Set Normalized without Cost matrix")

    confusion_matrix = helper.make_confusion_matrix(predictions_mc, test_data, test_indexes, False)
    data_plot.plot_confusion(confusion_matrix, root_dir, "MC dropout Test Set without Cost matrix")
    confusion_matrix = helper.confusion_array(confusion_matrix, dimension=1)
    data_plot.plot_confusion(confusion_matrix, root_dir, "MC dropout Test Set Normalized without Cost matrix")


#helper.find_lowest_cost([0.02939715244487161, 0.02633606596558821, 0.00489231509944943, 0.8639416721463203])

#SAVE_DIR = "best_model/"
#SAVE_DIR = "best_loss/"
#SAVE_DIR = "saved_models/Classifier 80 EPOCHs/best_model/"
#SAVE_DIR = "saved_model/"


#data_plot.plot_confusion(helper.get_cost_matrix(), SAVE_DIR, "My cost matrix")

#train_net(SAVE_DIR)
#helper.plot_samples(train_data, data_plot)

#network, optim, starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = load_net(SAVE_DIR, 8)
# network, optim, starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = load_net("saved_model/", 8)

#test(val_set, verbose=True)

"""train_net(SAVE_DIR,
          starting_epoch=starting_epoch,
          val_losses=val_losses,
          train_losses=train_losses,
          val_accuracies=val_accuracies,
          train_accuracies=train_accuracies)"""

if not os.path.exists("saved_models/"):
    os.mkdir("saved_models/")

for i in range(0, 10):
    
    network = model.Classifier(image_size, 8, class_weights, device, dropout=0.5, BBB=BBB)
    network.to(device)
    
    #optim = optimizer.Adam(network.parameters(), lr=0.001)
    
    #optim = optimizer.SGD(network.parameters(), lr=0.0001)
    #scheduler = optimizer.lr_scheduler.CyclicLR(optim, base_lr=0.0001, max_lr=0.01, step_size_up=10)
    
    optim = optimizer.SGD(network.parameters(), lr=0.00001, momentum=0.75)
    scheduler = optimizer.lr_scheduler.CyclicLR(optim, base_lr=0.00001, max_lr=0.01, step_size_up=10)
    
    print(optim)
    
    ROOT_SAVE_DIR = f"saved_models"
    #ROOT_SAVE_DIR = f"test_models"
    
    if not os.path.exists(ROOT_SAVE_DIR):
        os.mkdir(ROOT_SAVE_DIR)
        
    
    if BBB:
        ROOT_SAVE_DIR += f"/BBB_Classifier_{i}/"
    else:
        ROOT_SAVE_DIR += f"/Classifier_{i}/"
        
    train_net(ROOT_SAVE_DIR,
              starting_epoch=0,
              val_losses=[],
              train_losses=[],
              val_accuracies=[],
              train_accuracies=[],
              mc_val_losses=[],
              mc_val_accuracies=[])

    if BBB:

        SAVE_DIR = ROOT_SAVE_DIR + "best_BBB_loss/"
        network, optim, scheduler, starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = load_net(SAVE_DIR,
                                                                                                              8)

        if not os.path.exists(SAVE_DIR + "entropy/"):
            os.mkdir(SAVE_DIR + "entropy/")
            os.mkdir(SAVE_DIR + "variance/")
            os.mkdir(SAVE_DIR + "costs/")

        predictions_BBB_entropy, predictions_BBB_var, costs_BBB = testing.predict(test_set, SAVE_DIR, network,
                                                                                  test_size, device, BBB=True,
                                                                                  forward_passes=FORWARD_PASSES)
        helper.write_rows(predictions_BBB_entropy, SAVE_DIR + "BBB_entropy_predictions.csv")
        helper.write_rows(predictions_BBB_var, SAVE_DIR + "BBB_variance_predictions.csv")
        helper.write_rows(costs_BBB, SAVE_DIR + "BBB_costs.csv")

        costs_BBB = helper.read_rows(SAVE_DIR + "BBB_costs.csv")
        predictions_BBB = helper.read_rows(SAVE_DIR + "BBB_variance_predictions.csv")
        predictions_BBB = helper.read_rows(SAVE_DIR + "BBB_entropy_predictions.csv")


    if TRAIN_MC_DROPOUT:

        SAVE_DIR = ROOT_SAVE_DIR + "best_mc_loss/"
        network, optim, scheduler, starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = load_net(SAVE_DIR,
                                                                                                              8)

        if not os.path.exists(SAVE_DIR + "entropy/"):
            os.mkdir(SAVE_DIR + "entropy/")
            os.mkdir(SAVE_DIR + "variance/")
            os.mkdir(SAVE_DIR + "costs/")

        predictions_mc_entropy, predictions_mc_var, costs_mc = testing.predict(test_set, SAVE_DIR, network, test_size, device, mc_dropout=True, forward_passes=FORWARD_PASSES)
        helper.write_rows(predictions_mc_entropy, SAVE_DIR + "mc_entropy_predictions.csv")
        helper.write_rows(predictions_mc_var, SAVE_DIR + "mc_variance_predictions.csv")
        helper.write_rows(costs_mc, SAVE_DIR + "mc_costs.csv")

        predictions_mc = helper.read_rows(SAVE_DIR + "mc_entropy_predictions.csv")
        costs_mc = helper.read_rows(SAVE_DIR + "mc_costs.csv")

        #predictions_mc = helper.read_rows(SAVE_DIR + "mc_entropy_predictions.csv")
        #predictions_mc = helper.read_rows(SAVE_DIR + "mc_predictions.csv")

        predictions_mc = helper.string_to_float(predictions_mc)
        costs_mc = helper.string_to_float(costs_mc)

    if not BBB:

        SAVE_DIR = ROOT_SAVE_DIR + "best_loss/"
        network, optim, scheduler, starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = load_net(SAVE_DIR,
                                                                                                              8)

        if not os.path.exists(SAVE_DIR + "entropy/"):
            os.mkdir(SAVE_DIR + "entropy/")
            os.mkdir(SAVE_DIR + "variance/")
            os.mkdir(SAVE_DIR + "costs/")

        predictions_softmax, costs_softmax = testing.predict(test_set, SAVE_DIR, network, test_size, device,
                                                             softmax=True)
        helper.write_rows(predictions_softmax, SAVE_DIR + "softmax_predictions.csv")
        helper.write_rows(costs_softmax, SAVE_DIR + "softmax_costs.csv")

        predictions_softmax = helper.read_rows(SAVE_DIR + "softmax_predictions.csv")
        costs_sr = helper.read_rows(SAVE_DIR + "softmax_costs.csv")

        predictions_softmax = helper.string_to_float(predictions_softmax)
        costs_sr = helper.string_to_float(costs_sr)


SAVE_DIR = "saved_models/Classifier_0/best_mc_loss/"

predictions_mc = helper.read_rows(SAVE_DIR + "mc_entropy_predictions.csv")
costs_mc = helper.read_rows(SAVE_DIR + "mc_costs.csv")

SAVE_DIR = "saved_models/Classifier_0/best_loss/"

predictions_softmax = helper.read_rows(SAVE_DIR + "softmax_predictions.csv")
costs_sr = helper.read_rows(SAVE_DIR + "softmax_costs.csv")

SAVE_DIR = "saved_models/BBB_Classifier_0/best_BBB_loss/"

predictions_BBB = helper.read_rows(SAVE_DIR + "BBB_entropy_predictions.csv")
costs_BBB = helper.read_rows(SAVE_DIR + "BBB_costs.csv")

predictions_softmax = helper.string_to_float(predictions_softmax)
costs_sr = helper.string_to_float(costs_sr)
predictions_mc = helper.string_to_float(predictions_mc)
costs_mc = helper.string_to_float(costs_mc)


print_metrics(SAVE_DIR)


