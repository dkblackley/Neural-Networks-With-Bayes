"""
Main.py: The file responsible for being the entrypoint into the program,
Deals with things like weight balancing, training and testing methods and
calling other classes for plotting results of the network
"""

import sys
import torch
import torch.optim as optimizer
import numpy as np
import os

# Import other files
import testing
import training
import helper
import model
import constants

if torch.cuda.is_available():
    constants.ENABLE_GPU = True
else:
    constants.ENABLE_GPU = False

if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):

        if arg[0:2] == "-e":
            constants.EPOCHS = int(arg[2:])
        if arg[0:3] == "-fp":
            constants.FORWARD_PASSES = int(arg[3:])
        if arg[0:4] == "-bbb":
            constants.BBB = True
        if arg[0:5] == "-isic":
            constants.ISIC_pred = True
        if arg[0:4] == "-cpu":
            constants.ENABLE_GPU = False
        if arg[0:5] == "-load":
            constants.LOAD = True
        if arg[0:8] == "-predict":
            constants.TRAIN = False
        if arg[0:2] == "-n":
            constants.NUM_MODELS = int(arg[2:])

        print(f"Argument {i:>6}: {arg}")


def predict():
    constants.SAVE_DIR = "saved_models/SM_Classifier_0/"
    predictions_mc = helper.read_rows(constants.SAVE_DIR + "mc_entropy_predictions.csv")
    costs_mc = helper.read_rows(constants.SAVE_DIR + "mc_costs.csv")

    constants.SAVE_DIR = "saved_models/SM_Classifier_0/"
    predictions_softmax = helper.read_rows(constants.SAVE_DIR + "softmax_entropy.csv")
    costs_sr = helper.read_rows(constants.SAVE_DIR + "softmax_costs.csv")

    constants.SAVE_DIR = "saved_models/BBB_Classifier_0/"
    predictions_BBB = helper.read_rows(constants.SAVE_DIR + "BBB_entropy_predictions.csv")
    costs_BBB = helper.read_rows(constants.SAVE_DIR + "BBB_costs.csv")

    predictions_softmax = helper.string_to_float(predictions_softmax)
    costs_sr = helper.string_to_float(costs_sr)
    predictions_mc = helper.string_to_float(predictions_mc)
    costs_mc = helper.string_to_float(costs_mc)
    predictions_BBB = helper.string_to_float(predictions_BBB)
    costs_BBB = helper.string_to_float(costs_BBB)

    constants.SAVE_DIR = "saved_models/images/"

    if not os.path.exists(constants.SAVE_DIR):
        os.mkdir(constants.SAVE_DIR)

    data_plot.print_metrics(constants.SAVE_DIR, costs_sr, costs_mc, costs_BBB, predictions_softmax, predictions_mc, predictions_BBB, test_data)

if __name__ == "main":

    np.random.seed(1337)
    for i in range(0, constants.NUM_MODELS):
        training.setup()
        training.train()
        predict()

def print_metrics():
    pass
