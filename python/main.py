"""
Main.py: The file responsible for being the entrypoint into the program,
Deals with things like weight balancing, training and testing methods and
calling other classes for plotting results of the network
"""

import torch
import torch.optim as optimizer
from torchvision import transforms
from torch.utils.data import random_split, SubsetRandomSampler, Subset
import numpy as np
import data_loading
import data_plotting
import testing
import helper
import model
import torch.nn as nn
from tqdm import tqdm

LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC', 8: 'UNK'}
EPOCHS = 0
UNKNOWN_CLASS = False
DEBUG = False  # Toggle this to only run for 1% of the training data
ENABLE_GPU = False  # Toggle this to enable or disable GPU
BATCH_SIZE = 32
SOFTMAX = True
MC_DROPOUT = False
COST_MATRIX = False
TEST_COST_MATRIX = False
FORWARD_PASSES = 100
BBB = False
image_size = 224
test_indexes = []
test_size = 0
val_size = 0
train_size = 0
best_val = 0
best_loss = 0

if ENABLE_GPU:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")



composed_train = transforms.Compose([
                                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                                transforms.ColorJitter(brightness=0.2),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomAffine(0, shear=0.01),
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
        split_val = int(np.floor(0.33 * split_train))

        np.random.seed(1337)
        np.random.shuffle(indices)

        temp_idx, train_idx = indices[split_train:], indices[:split_train]
        valid_idx, test_idx = temp_idx[split_val:], temp_idx[:split_val]

        for i in scc_idx:
            if i not in test_idx:
                test_idx.append(i)

        np.random.shuffle(test_idx)

    else:
        indices = list(range(len(train_data)))
        split_train = int(np.floor(0.7 * len(train_data)))
        split_val = int(np.floor(0.33 * split_train))

        np.random.seed(1337)
        np.random.shuffle(indices)

        temp_idx, train_idx = indices[split_train:], indices[:split_train]
        valid_idx, test_idx = temp_idx[split_val:], temp_idx[:split_val]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # Don't shuffle the testing set for MC_DROPOUT
    testing_data = Subset(test_data, test_idx)
    #test_sampler = SequentialSampler(test_temp)

    training_set = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)
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

if UNKNOWN_CLASS:
    network = model.Classifier(image_size, 7, dropout=0.5)
    network.to(device)
    weights = [3188, 8985, 2319, 602, 1862, 164, 170]
else:
    network = model.Classifier(image_size, 8, dropout=0.5)
    network.to(device)
    weights = [3188, 8985, 2319, 602, 1862, 164, 170, 441]

optim = optimizer.Adam(network.parameters(), lr=0.001)

#weights = list(helper.count_classes(train_set, BATCH_SIZE).values())


new_weights = []
index = 0
for weight in weights:
    if UNKNOWN_CLASS:
        new_weights.append(sum(weights) / (7 * weight))
    else:
        new_weights.append(sum(weights)/(8 * weight))
    index = index + 1

#weights = [4522, 12875, 3323, 867, 2624, 239, 253, 628]

#new_weights = helper.apply_cost_matrix()

class_weights = torch.Tensor(new_weights).to(device)
if COST_MATRIX:
    loss_function = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
else:
    loss_function = nn.CrossEntropyLoss(weight=class_weights)


def train(current_epoch, val_losses, train_losses, val_accuracy, train_accuracy, verbose=False):
    """
    trains the network while also recording the accuracy of the network on the training data
    :param verboose: If true dumps out debug info about which classes the network is predicting when correct and incorrect
    :return: returns the number of epochs, a list of the validation set losses per epoch, a list of the
    training set losses per epoch, a list of the validation accuracy per epoch and the
    training accuracy per epoch
    """

    input("\nHave you remembered to move the model out of saved model before loading in the best model?")

    intervals = []
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

    print("\nTraining Network")

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
            image_batch = sample_batch['image']
            label_batch = sample_batch['label']

            image_batch, label_batch = image_batch.to(device), label_batch.to(device)

            optim.zero_grad()
            outputs = network(image_batch, dropout=True)


            if COST_MATRIX:
                loss_values = loss_function(outputs, label_batch)
                temp_loss = torch.tensor(0.0)
                weighting = torch.tensor(0.0)
                i = 0
                for loss_value in loss_values:
                    cost = helper.get_cost(torch.argmax(outputs[i]), label_batch[i])
                    if new_weights:
                        weighting = weighting + new_weights[label_batch[i]]
                    else:
                        weighting = torch.tensor(1.0)
                    loss_value = loss_value * cost
                    temp_loss = temp_loss + loss_value
                    i = i + 1

                loss = temp_loss/(weighting)
            else:
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

        save_network(optim, val_losses, train_losses, val_accuracy, train_accuracy, "saved_model/")

        if best_val < max(val_accuracy):
            save_network(optim, val_losses, train_losses, val_accuracy, train_accuracy, "best_model/")
            best_val = max(val_accuracy)

        if best_loss > min(val_losses):
            save_network(optim, val_losses, train_losses, val_accuracy, train_accuracy, "best_loss/")
            best_loss = min(val_losses)

    data_plot.plot_loss(intervals, val_losses, train_losses)
    data_plot.plot_validation(intervals, val_accuracy, train_accuracy)
    save_network(optim, val_losses, train_losses, val_accuracy, train_accuracy, "saved_model/")

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
    correct_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}
    incorrect_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}
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

                if COST_MATRIX:
                    loss_values = loss_function(outputs, label_batch)
                    temp_loss = torch.tensor(0.0)
                    weighting = torch.tensor(0.0)
                    i = 0
                    for loss_value in loss_values:
                        cost = helper.get_cost(torch.argmax(outputs[i]), label_batch[i])
                        weighting = weighting + new_weights[label_batch[i]]
                        loss_value = loss_value * cost
                        temp_loss = temp_loss + loss_value
                        i = i + 1

                    loss = temp_loss / (weighting)
                else:
                    loss = loss_function(outputs, label_batch)

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

def save_network(optim, val_losses, train_losses, val_accuracies, train_accuracies, root_dir):
    helper.save_net(network, optim, root_dir + "model_parameters")
    helper.write_csv(val_losses, root_dir + "val_losses.csv")
    helper.write_csv(train_losses, root_dir + "train_losses.csv")
    helper.write_csv(val_accuracies, root_dir + "val_accuracies.csv")
    helper.write_csv(train_accuracies, root_dir + "train_accuracies.csv")


def load_net(root_dir, output_size):

    val_losses = helper.read_csv(root_dir + "val_losses.csv")
    train_losses = helper.read_csv(root_dir + "train_losses.csv")
    val_accuracies = helper.read_csv(root_dir + "val_accuracies.csv")
    train_accuracies = helper.read_csv(root_dir + "train_accuracies.csv")
    network, optim = helper.load_net(root_dir + "model_parameters", image_size, output_size)

    return network, optim, len(train_losses), val_losses, train_losses, val_accuracies, train_accuracies


def train_net(root_dir, starting_epoch=0, val_losses=[], train_losses=[], val_accuracies=[], train_accuracies=[]):
    """
    Trains a network, saving the parameters and the losses/accuracies over time
    :return:
    """
    starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = train(
        starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies, verbose=True)

    if not DEBUG:

        if UNKNOWN_CLASS:
            # could replace with a for loop
            predictions_sr = testing.predict(val_set, network, val_size, softmax=True)
            confusion_matrix = helper.predictions_to_confusion()
            data_plot.plot_confusion(confusion_matrix, root_dir, "Validation Set")
            confusion_matrix = helper.confusion_array(confusion_matrix)
            data_plot.plot_confusion(confusion_matrix, root_dir, "Validation Set Normalized")

            _, __, confusion_matrix = test(train_set, verbose=True)
            data_plot.plot_confusion(confusion_matrix, root_dir, "Training Set")
            confusion_matrix = helper.confusion_array(confusion_matrix)
            data_plot.plot_confusion(confusion_matrix, root_dir, "Training Set Normalized")

            _, __, confusion_matrix = test(test_set, verbose=True)
            data_plot.plot_confusion(confusion_matrix, root_dir, "Testing Set")
            confusion_matrix = helper.confusion_array(confusion_matrix)
            data_plot.plot_confusion(confusion_matrix, root_dir, "Testing Set Normalized")
        else:
            _, __, confusion_matrix = test(val_set, verbose=True)
            data_plot.plot_confusion(confusion_matrix, root_dir, "Validation Set")
            confusion_matrix = helper.confusion_array(confusion_matrix)
            data_plot.plot_confusion(confusion_matrix, root_dir, "Validation Set Normalized")

            _, __, confusion_matrix = test(train_set, verbose=True)
            data_plot.plot_confusion(confusion_matrix, root_dir, "Training Set")
            confusion_matrix = helper.confusion_array(confusion_matrix)
            data_plot.plot_confusion(confusion_matrix, root_dir, "Training Set Normalized")

            _, __, confusion_matrix = test(test_set, verbose=True)
            data_plot.plot_confusion(confusion_matrix, root_dir, "Testing Set")
            confusion_matrix = helper.confusion_array(confusion_matrix)
            data_plot.plot_confusion(confusion_matrix, root_dir, "Testing Set Normalized")


    return val_losses, train_losses, val_accuracies, train_accuracies


def print_metrics(model_name):

    data_plot.plot_true_cost_coverage_by_class([predictions_mc, predictions_softmax], model_name,
                                               "Average Test cost by Classes using raw Probabilities with Flattened Matrix",
                                               costs=False, flatten=True)

    data_plot.plot_true_cost_coverage_by_class([costs_mc, costs_sr], model_name,
                                      "Average Test cost by Classes using LEC")

    data_plot.plot_true_cost_coverage_by_class([predictions_mc, predictions_softmax], model_name,
                                               "Average Test cost by Classes using raw Probabilities", costs=False)



    data_plot.plot_true_cost_coverage(predictions_mc, predictions_softmax, model_name,
                                      "Average Test cost using Raw Probabilities with Flattened Matrix", costs=False, flatten=True)
    data_plot.plot_true_cost_coverage(predictions_mc, predictions_softmax, model_name,
                                      "Average Test cost using Raw Probabilities", costs=False)
    data_plot.plot_true_cost_coverage(costs_mc, costs_sr, model_name, "Average Test cost using LEC")

    data_plot.plot_cost_coverage(costs_mc, costs_sr, model_name, "Coverage by Lowest Expected cost", load=False)

    data_plot.count_sampels_in_intervals(predictions_mc, model_name, "Number of Samples in each Interval MC Dropout", 5)
    data_plot.count_sampels_in_intervals(predictions_mc, model_name, "Number of Samples in each Interval MC Dropout (Without probabilites below 0.2)", 5, skip_first=True)
    data_plot.count_sampels_in_intervals(predictions_softmax, model_name, "Number of Samples in each Interval Softmax", 5)
    data_plot.count_sampels_in_intervals(predictions_softmax, model_name, "Number of Samples in each Interval Softmax (Without probabilites below 0.2)", 5, skip_first=True)
    data_plot.plot_each_mc_cost(predictions_softmax, model_name, "Costs by Forward Pass", "entropy")



    data_plot.plot_calibration(predictions_mc, "MC Dropout Reliability Diagram", model_name, 5)
    data_plot.plot_calibration(predictions_softmax, "Softmax Reliability Diagram", model_name, 5)

    costs_with_entropy_mc = helper.attach_last_row(costs_mc, predictions_mc)
    costs_with_entropy_sr = helper.attach_last_row(costs_sr, predictions_softmax)
    data_plot.plot_cost_coverage(costs_with_entropy_mc, costs_with_entropy_sr, model_name, "Risk Coverage by Uncertainty", uncertainty=True)




    correct_mc, incorrect_mc, uncertain_mc = helper.get_correct_incorrect(predictions_mc, test_data, test_indexes, TEST_COST_MATRIX)
    print(f"MC Accuracy: {len(correct_mc)/(len(correct_mc) + len(incorrect_mc)) * 100}")

    correct_sr, incorrect_sr, uncertain_sr = helper.get_correct_incorrect(predictions_softmax, test_data, test_indexes, TEST_COST_MATRIX)
    print(f"SM Accuracy: {len(correct_sr) / (len(correct_sr) + len(incorrect_sr)) * 100}")

    data_plot.plot_correct_incorrect_uncertainties(correct_mc, incorrect_mc, model_name, "MC Dropout Variance by class", by_class=True, prediction_index=0)
    data_plot.plot_correct_incorrect_uncertainties(correct_sr, incorrect_sr, model_name, "Softmax Response Variance by class", by_class=True, prediction_index=0)

    data_plot.plot_correct_incorrect_uncertainties(correct_mc, incorrect_mc, model_name, "MC Dropout Variance across predictions")
    data_plot.plot_correct_incorrect_uncertainties(correct_sr, incorrect_sr, model_name, "Softmax Response Entropies across predictions")

    data_plot.average_uncertainty_by_class(correct_mc, incorrect_mc, model_name, "MC Dropout Accuracies by Prediction")
    data_plot.average_uncertainty_by_class(correct_sr, incorrect_sr, model_name, "Softmax Response Accuracies by Prediction")



    data_plot.plot_risk_coverage(predictions_mc, predictions_softmax, model_name, "Risk Coverage", load=False)


    confusion_matrix = helper.make_confusion_matrix(predictions_softmax, test_data, test_indexes, True)
    data_plot.plot_confusion(confusion_matrix, model_name, "Softmax Response on Test Set with Cost matrix")
    confusion_matrix = helper.confusion_array(confusion_matrix, dimension=1)
    data_plot.plot_confusion(confusion_matrix, model_name, "Softmax Response Test Set Normalized with Cost matrix")

    confusion_matrix = helper.make_confusion_matrix(predictions_mc, test_data, test_indexes, True)
    data_plot.plot_confusion(confusion_matrix, model_name, "MC dropout Test Set with Cost matrix")
    confusion_matrix = helper.confusion_array(confusion_matrix, dimension=1)
    data_plot.plot_confusion(confusion_matrix, model_name, "MC dropout Test Set Normalized with Cost matrix")

    confusion_matrix = helper.make_confusion_matrix(predictions_softmax, test_data, test_indexes, False)
    data_plot.plot_confusion(confusion_matrix, model_name, "Softmax Response on Test Set without Cost matrix")
    confusion_matrix = helper.confusion_array(confusion_matrix, dimension=1)
    data_plot.plot_confusion(confusion_matrix, model_name, "Softmax Response Test Set Normalized without Cost matrix")

    confusion_matrix = helper.make_confusion_matrix(predictions_mc, test_data, test_indexes, False)
    data_plot.plot_confusion(confusion_matrix, model_name, "MC dropout Test Set without Cost matrix")
    confusion_matrix = helper.confusion_array(confusion_matrix, dimension=1)
    data_plot.plot_confusion(confusion_matrix, model_name, "MC dropout Test Set Normalized without Cost matrix")


#helper.find_lowest_cost([0.02939715244487161, 0.02633606596558821, 0.00489231509944943, 0.8639416721463203])

#model_name = "best_model/"
#model_name = "best_loss/"
model_name = "saved_models/Classifier 80 EPOCHs/best_model/"
#model_name = "saved_model/"


#data_plot.plot_confusion(helper.get_cost_matrix(), model_name, "My cost matrix")

#train_net(model_name)
#helper.plot_samples(train_data, data_plot)

network, optim, starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = load_net(model_name, 8)
# network, optim, starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = load_net("saved_model/", 8)

#test(val_set, verbose=True)

"""train_net(model_name,
          starting_epoch=starting_epoch,
          val_losses=val_losses,
          train_losses=train_losses,
          val_accuracies=val_accuracies,
          train_accuracies=train_accuracies)"""


"""predictions_mc_entropy, predictions_mc_var, costs_mc = testing.predict(test_set, model_name, network, test_size, mc_dropout=True, forward_passes=FORWARD_PASSES)
helper.write_rows(predictions_mc_entropy, model_name + "mc_entropy_predictions.csv")
helper.write_rows(predictions_mc_var, model_name + "mc_variance_predictions.csv")
helper.write_rows(costs_mc, model_name + "mc_costs.csv")

predictions_softmax, costs_softmax = testing.predict(test_set, model_name, network, test_size, softmax=True)
helper.write_rows(predictions_softmax, model_name + "softmax_predictions.csv")
helper.write_rows(costs_softmax, model_name + "softmax_costs.csv")"""


predictions_softmax = helper.read_rows(model_name + "softmax_predictions.csv")
predictions_mc = helper.read_rows(model_name + "mc_entropy_predictions.csv")

costs_mc = helper.read_rows(model_name + "costs/mc_forward_pass_99_costs.csv")
costs_sr = helper.read_rows(model_name + "softmax_costs.csv")
#predictions_mc = helper.read_rows(model_name + "mc_entropy_predictions.csv")
#predictions_mc = helper.read_rows(model_name + "mc_predictions.csv")

predictions_softmax = helper.string_to_float(predictions_softmax)
predictions_mc = helper.string_to_float(predictions_mc)

costs_mc = helper.string_to_float(costs_mc)
costs_sr = helper.string_to_float(costs_sr)

print_metrics(model_name)


