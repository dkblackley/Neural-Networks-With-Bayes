"""
Main.py: The file responsible for being the entrypoint into the program,
Deals with things like weight balancing, training and testing methods and
calling other classes for plotting results of the network
"""

# Global Variables
EPOCHS = 50
DEBUG = False # Toggle this to only run for 1% of the training data
ENABLE_GPU = True  # Toggle this to enable or disable GPU
BATCH_SIZE = 16
SOFTMAX = True
TRAIN_MC_DROPOUT = False
SAMPLES = 3
FORWARD_PASSES = 100
BBB = True
LOAD = False
LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC', 8: 'UNK'}
SAVE_DIR = "saved_models"
ISIC_pred = False
TRAIN = True
NUM_MODELS = 1

composed_train = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(35),
    transforms.Resize(int(image_size * 1.5)),
    transforms.CenterCrop(int(image_size * 1.25)),
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

import sys
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

if torch.cuda.is_available():
    ENABLE_GPU = True
else:
    ENABLE_GPU = False

if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):

        if arg[0:2] == "-e":
            EPOCHS = int(arg[2:])
        if arg[0:3] == "-fp":
            FORWARD_PASSES = int(arg[3:])
        if arg[0:4] == "-bbb":
            BBB = True
        if arg[0:5] == "-isic":
            ISIC_pred = True
        if arg[0:4] == "-cpu":
            ENABLE_GPU = False
        if arg[0:5] == "-load":
            LOAD = True
        if arg[0:8] == "-predict":
            TRAIN = False
        if arg[0:2] == "-n":
            NUM_MODELS = int(arg[2:])

        print(f"Argument {i:>6}: {arg}")


def setup():
    image_size = 224
    np.random.seed(1337)

    if ENABLE_GPU:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")



    weights = [3188, 8985, 2319, 602, 1862, 164, 170, 441] # Distribution when using 70% of dataset

    # Calculate the weights for the sampler function and loss functions
    new_weights = []
    sampler_weights = []
    val_weights = []
    k = 0
    q = 1

    for weight in weights:
        new_weights.append(((sum(weights))/weight)**k)
        sampler_weights.append(((sum(weights))/weight)**q)
        val_weights.append(((sum(weights))/weight)**1)

    class_weights = torch.Tensor(new_weights)/new_weights[1]
    sampler_weights = torch.Tensor(sampler_weights)/new_weights[1]
    val_weights = torch.Tensor(val_weights)/new_weights[1]

    val_weights = val_weights.to(device)
    class_weights = class_weights.to(device)
    sampler_weights = sampler_weights.to(device)


    SAVE_DIR = "saved_models/SM_Classifier_0/"
    predictions_mc = helper.read_rows(SAVE_DIR + "mc_entropy_predictions.csv")
    costs_mc = helper.read_rows(SAVE_DIR + "mc_costs.csv")

    SAVE_DIR = "saved_models/SM_Classifier_0/"

    predictions_softmax = helper.read_rows(SAVE_DIR + "softmax_entropy.csv")
    #predictions_softmax = helper.read_rows(SAVE_DIR + "softmax_predictions.csv")
    costs_sr = helper.read_rows(SAVE_DIR + "softmax_costs.csv")

    SAVE_DIR = "saved_models/BBB_Classifier_0/"

    predictions_BBB = helper.read_rows(SAVE_DIR + "BBB_entropy_predictions.csv")
    costs_BBB = helper.read_rows(SAVE_DIR + "BBB_costs.csv")

    predictions_softmax = helper.string_to_float(predictions_softmax)
    costs_sr = helper.string_to_float(costs_sr)
    predictions_mc = helper.string_to_float(predictions_mc)
    costs_mc = helper.string_to_float(costs_mc)
    predictions_BBB = helper.string_to_float(predictions_BBB)
    costs_BBB = helper.string_to_float(costs_BBB)

    SAVE_DIR = "saved_models/images/"

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    data_plot.print_metrics(SAVE_DIR, costs_sr, costs_mc, costs_BBB, predictions_softmax, predictions_mc, predictions_BBB, test_data)

if __name__ == "main":
    for i in range(0, NUM_MODELS):

        network = model.Classifier(image_size, 8, class_weights, device, dropout=0.5, BBB=BBB)
        network.to(device)

        if BBB:

            # Set the learning rate to be higher for the Bayesian Layer
            BBB_weights = ['hidden_layer.weight_mu', 'hidden_layer.weight_rho', 'hidden_layer.bias_mu',
                           'hidden_layer.bias_rho',
                           'hidden_layer2.weight_mu', 'hidden_layer2.weight_rho', 'hidden_layer2.bias_mu',
                           'hidden_layer2.bias_rho']

            BBB_parameters = list(
                map(lambda x: x[1], list(filter(lambda kv: kv[0] in BBB_weights, network.named_parameters()))))
            base_parameters = list(
                map(lambda x: x[1], list(filter(lambda kv: kv[0] not in BBB_weights, network.named_parameters()))))

            optim = optimizer.SGD([
                {'params': BBB_parameters},
                {'params': base_parameters, 'lr': 0.0001}
            ], lr=0.0001, momentum=0.9, weight_decay=0.00001)

            scheduler = optimizer.lr_scheduler.CyclicLR(optim, base_lr=[0.0001, 0.0001], max_lr=[0.1, 0.02],
                                                        step_size_up=(555 * 5), mode="triangular2")

        else:
            optim = optimizer.SGD(network.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.00001)
            scheduler = optimizer.lr_scheduler.CyclicLR(optim, base_lr=0.0001, max_lr=0.02, step_size_up=(555 * 5),
                                                        mode="triangular2")

        ROOT_SAVE_DIR = SAVE_DIR

        if not os.path.exists(ROOT_SAVE_DIR):
            os.mkdir(ROOT_SAVE_DIR)

        if BBB:
            SAVE_DIR += f"/BBB_Classifier_{i}/"
        elif TRAIN_MC_DROPOUT:
            SAVE_DIR += f"/MC_Classifier_{i}/"
        else:
            SAVE_DIR += f"/SM_Classifier_{i}/"

        if LOAD:
            network, optim, scheduler, starting_epoch, \
                val_losses, train_losses, val_accuracies, \
                train_accuracies = helper.load_net(SAVE_DIR, 8, image_size, device, class_weights)
            if TRAIN:
                starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = train(SAVE_DIR,
                                                                                                   starting_epoch,
                                                                                                   val_losses,
                                                                                                   train_losses,
                                                                                                   val_accuracies,
                                                                                                   train_accuracies,
                                                                                                   verbose=False)

        else:

            starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = train(SAVE_DIR, 0,
                                                                                               [], [],
                                                                                               [], [],
                                                                                               verbose=True)

        if BBB:
            network, optim, scheduler, starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = helper.load_net(
                SAVE_DIR, 8, image_size, device, class_weights)

            if not os.path.exists(SAVE_DIR + "entropy/"):
                os.mkdir(SAVE_DIR + "entropy/")
                os.mkdir(SAVE_DIR + "variance/")
                os.mkdir(SAVE_DIR + "costs/")

            if ISIC_pred:
                predictions_BBB_entropy, predictions_BBB_var, costs_BBB = testing.predict(ISIC_set, SAVE_DIR, network,
                                                                                          len(ISIC_data),
                                                                                          device, mc_dropout=True,
                                                                                          forward_passes=FORWARD_PASSES,
                                                                                          ISIC=True)
                predictions_BBB_entropy.insert(0,
                                               ["image", "MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"])
            else:

                predictions_BBB_entropy, predictions_BBB_var, costs_BBB = testing.predict(test_set, SAVE_DIR, network,
                                                                                          test_size, device, BBB=True,
                                                                                          forward_passes=FORWARD_PASSES)
            helper.write_rows(predictions_BBB_entropy, SAVE_DIR + "BBB_entropy_predictions.csv")
            helper.write_rows(predictions_BBB_var, SAVE_DIR + "BBB_variance_predictions.csv")
            helper.write_rows(costs_BBB, SAVE_DIR + "BBB_costs.csv")

            costs_BBB = helper.read_rows(SAVE_DIR + "BBB_costs.csv")
            predictions_BBB = helper.read_rows(SAVE_DIR + "BBB_entropy_predictions.csv")

        if SOFTMAX:

            network, optim, scheduler, starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = helper.load_net(
                SAVE_DIR, 8, image_size, device, class_weights)

            if not os.path.exists(SAVE_DIR + "entropy/"):
                os.mkdir(SAVE_DIR + "entropy/")
                os.mkdir(SAVE_DIR + "variance/")
                os.mkdir(SAVE_DIR + "costs/")

            if ISIC_pred:
                predictions_softmax, entropy_soft, costs_softmax = testing.predict(ISIC_set, SAVE_DIR, network,
                                                                                   len(ISIC_data), device,
                                                                                   softmax=True, ISIC=True)
                predictions_softmax.insert(0, ["image", "MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"])
            else:
                predictions_softmax, entropy_soft, costs_softmax = testing.predict(test_set, SAVE_DIR, network,
                                                                                   test_size, device,
                                                                                   softmax=True)
            helper.write_rows(predictions_softmax, SAVE_DIR + "softmax_predictions.csv")
            helper.write_rows(entropy_soft, SAVE_DIR + "softmax_entropy.csv")
            helper.write_rows(costs_softmax, SAVE_DIR + "softmax_costs.csv")

            predictions_softmax = helper.read_rows(SAVE_DIR + "softmax_predictions.csv")
            costs_sr = helper.read_rows(SAVE_DIR + "softmax_costs.csv")

            if ISIC_pred:
                predictions_mc_entropy, predictions_mc_var, costs_mc = testing.predict(ISIC_set, SAVE_DIR, network,
                                                                                       len(ISIC_data),
                                                                                       device, mc_dropout=True,
                                                                                       forward_passes=FORWARD_PASSES,
                                                                                       ISIC=True)
                predictions_mc_entropy.insert(0, ["image", "MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"])
            else:
                predictions_mc_entropy, predictions_mc_var, costs_mc = testing.predict(test_set, SAVE_DIR, network,
                                                                                       test_size,
                                                                                       device, mc_dropout=True,
                                                                                       forward_passes=FORWARD_PASSES)
            helper.write_rows(predictions_mc_entropy, SAVE_DIR + "mc_entropy_predictions.csv")
            helper.write_rows(predictions_mc_var, SAVE_DIR + "mc_variance_predictions.csv")
            helper.write_rows(costs_mc, SAVE_DIR + "mc_costs.csv")

            predictions_mc = helper.read_rows(SAVE_DIR + "mc_entropy_predictions.csv")
            costs_mc = helper.read_rows(SAVE_DIR + "mc_costs.csv")

