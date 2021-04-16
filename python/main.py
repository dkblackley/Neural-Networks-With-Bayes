"""
Main.py: The file responsible for being the entrypoint into the program,
Deals with things like weight balancing, training and testing methods and
calling other classes for plotting results of the network
"""

# Global Variables
EPOCHS = 3
DEBUG = True # Toggle this to only run for 1% of the training data
ENABLE_GPU = False  # Toggle this to enable or disable GPU
BATCH_SIZE = 32
SOFTMAX = True
TRAIN_MC_DROPOUT = False
SAMPLES = 3
FORWARD_PASSES = 3
BBB = False
LOAD = True
LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC', 8: 'UNK'}
SAVE_DIR = "saved_models"
ISIC_pred = True

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
ISIC_data = data_loading.data_set("Training_meta_data/ISIC_2019_Training_Metadata.csv", "ISIC_2019_Test_Input",  transforms=composed_test)

def get_data_sets(plot=False):
    """
    Splits the data sets into train, test and validation sets
    :param plot: If true, plot some samples form each set
    :return: the DataLoader objects, ready to be called from for each set
    """

    # 70% split to train set, use 2/3rds of the remaining data (20%) for the testing set
    indices = list(range(len(train_data)))
    split_train = int(np.floor(0.7 * len(indices)))
    split_test = int(np.floor(0.6667 * (len(indices) - split_train)))

    temp_idx, train_idx = indices[split_train:], indices[:split_train]
    valid_idx, test_idx = temp_idx[split_test:], temp_idx[:split_test]

    weighted_train_idx = []

    for c in range(0, len(train_idx)):
        label = train_data.get_label(train_idx[c])
        weighted_idx = sampler_weights[label]
        weighted_train_idx.append(weighted_idx)

    weighted_train_sampler = WeightedRandomSampler(weights=weighted_train_idx, num_samples=len(weighted_train_idx), replacement=True)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Don't shuffle the testing set for MC_DROPOUT
    testing_data = Subset(test_data, test_idx)

    training_set = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=weighted_train_sampler, shuffle=False)
    valid_set = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, sampler=valid_sampler)
    testing_set = torch.utils.data.DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=False)
    ISIC_set = torch.utils.data.DataLoader(ISIC_data, batch_size=BATCH_SIZE, shuffle=False)

    if plot:

        data_plot = data_plotting.DataPlotting(test_data, test_idx)

        helper.plot_set(training_set, data_plot, 0, 5)
        helper.plot_set(valid_set, data_plot, 0, 5)
        helper.plot_set(testing_set, data_plot, 0, 5)
        helper.plot_set(ISIC_set, data_plot, 0, 5)

    return training_set, valid_set, testing_set, ISIC_set, len(test_idx), len(train_idx), len(valid_idx), test_idx

train_set, val_set, test_set, ISIC_set, test_size, train_size, val_size, test_indexes = get_data_sets(plot=True)

data_plot = data_plotting.DataPlotting(test_data, test_indexes)
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


"""def train_net(root_dir, starting_epoch=0, val_losses=[], train_losses=[], val_accuracies=[], train_accuracies=[]):
    starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = train(root_dir, starting_epoch,
                                                                                       val_losses, train_losses,
                                                                                       val_accuracies, train_accuracies,
                                                                                       verbose=True)

    return val_losses, train_losses, val_accuracies, train_accuracies"""

                       
def print_metrics(root_dir):
    helper.remove_last_row(costs_sr)
    helper.remove_last_row(costs_mc)
    helper.remove_last_row(costs_BBB)

    data_plot.plot_each_mc_true_cost(costs_sr, root_dir, "saved_models/SM_Classifier_0/",
                                     "saved_models/BBB_Classifier_0/", "Test cost by forward pass")

    data_plot.plot_each_mc_pass("saved_models/SM_Classifier_0/",
                                "saved_models/BBB_Classifier_0/", predictions_softmax, test_indexes,
                                test_data, root_dir, "Accuracies by Forward Pass", cost_matrix=False)

    predictions = [predictions_softmax, predictions_mc, predictions_BBB]
    costs = [costs_sr, costs_mc, costs_BBB]

    data_plot.plot_true_cost_coverage_by_class(costs, root_dir,
                                               "Average Test cost by Classes using LEC")
    data_plot.plot_true_cost_coverage(costs, root_dir, "Coverage by Average Test cost", uncertainty=False)

    data_plot.plot_true_cost_coverage_by_class(predictions, root_dir,
                                               "Average Test cost by Classes using raw Probabilities with Flattened Matrix",
                                               costs=False, flatten=True)

    data_plot.plot_true_cost_coverage_by_class(costs, root_dir,
                                      "Average Test cost by Classes using LEC")

    data_plot.plot_true_cost_coverage_by_class(predictions, root_dir,
                                               "Average Test cost by Classes using raw Probabilities", costs=False)



    data_plot.plot_true_cost_coverage(predictions, root_dir,
                                      "Average Test cost using Raw Probabilities with Flattened Matrix", costs=False, flatten=True, uncertainty=True)
    data_plot.plot_true_cost_coverage(predictions, root_dir,
                                      "Average Test cost using Raw Probabilities", costs=False)
    data_plot.plot_true_cost_coverage(costs, root_dir, "Average Test cost using LEC")

    data_plot.plot_cost_coverage(costs, root_dir, "Coverage by Lowest Expected cost")

    data_plot.count_sampels_in_intervals(predictions_mc, root_dir, "Number of Samples in each Interval MC Dropout", 5)
    data_plot.count_sampels_in_intervals(predictions_mc, root_dir, "Number of Samples in each Interval MC Dropout (Without probabilites below 0.2)", 5, skip_first=True)

    data_plot.count_sampels_in_intervals(predictions_softmax, root_dir, "Number of Samples in each Interval Softmax", 5)
    data_plot.count_sampels_in_intervals(predictions_softmax, root_dir,
                                         "Number of Samples in each Interval Softmax (Without probabilites below 0.2)",
                                         5, skip_first=True)

    data_plot.count_sampels_in_intervals(predictions_BBB, root_dir, "Number of Samples in each Interval BbB", 5)
    data_plot.count_sampels_in_intervals(predictions_BBB, root_dir,
                                         "Number of Samples in each Interval BbB (Without probabilites below 0.2)",
                                         5, skip_first=True)


    data_plot.plot_calibration(predictions_mc, "MC Dropout Reliability Diagram", root_dir, 5)
    data_plot.plot_calibration(predictions_softmax, "Softmax Reliability Diagram", root_dir, 5)
    data_plot.plot_calibration(predictions_BBB, "BbB Reliability Diagram", root_dir, 5)

    costs_with_entropy_mc = helper.attach_last_row(costs_mc, predictions_mc)
    costs_with_entropy_sr = helper.attach_last_row(costs_sr, predictions_softmax)
    costs_with_entropy_BBB = helper.attach_last_row(costs_BBB, predictions_BBB)

    costs_with_entropy = [costs_with_entropy_sr, costs_with_entropy_mc, costs_with_entropy_BBB]

    data_plot.plot_cost_coverage(costs_with_entropy, root_dir, "Risk Coverage by Uncertainty", uncertainty=False)

    data_plot.plot_risk_coverage(predictions, root_dir, "Risk Coverage")

    correct_mc, incorrect_mc, uncertain_mc = helper.get_correct_incorrect(predictions_mc, test_data, test_indexes,
                                                                          False)
    print(f"MC Accuracy: {len(correct_mc)/(len(correct_mc) + len(incorrect_mc)) * 100}")

    correct_sr, incorrect_sr, uncertain_sr = helper.get_correct_incorrect(predictions_softmax, test_data, test_indexes,
                                                                          False)
    print(f"SM Accuracy: {len(correct_sr) / (len(correct_sr) + len(incorrect_sr)) * 100}")

    correct_BBB, incorrect_BBB, uncertain_BBB = helper.get_correct_incorrect(predictions_BBB, test_data, test_indexes,
                                                                          False)
    print(f"BbB Accuracy: {len(correct_BBB) / (len(correct_BBB) + len(incorrect_BBB)) * 100}")

    data_plot.plot_correct_incorrect_uncertainties(correct_mc, incorrect_mc, root_dir, "MC Dropout Entropy by class", by_class=True, prediction_index=0)
    data_plot.plot_correct_incorrect_uncertainties(correct_sr, incorrect_sr, root_dir, "Softmax Response Variance by class", by_class=True, prediction_index=0)
    data_plot.plot_correct_incorrect_uncertainties(correct_BBB, incorrect_BBB, root_dir, "BbB Entropy by class", by_class=True, prediction_index=0)

    data_plot.plot_correct_incorrect_uncertainties(correct_mc, incorrect_mc, root_dir, "MC Dropout Entropy across predictions")
    data_plot.plot_correct_incorrect_uncertainties(correct_sr, incorrect_sr, root_dir, "Softmax Response Entropies across predictions")
    data_plot.plot_correct_incorrect_uncertainties(correct_BBB, incorrect_BBB, root_dir, "BbB Response Entropies across predictions")

    data_plot.average_uncertainty_by_class(correct_mc, incorrect_mc, root_dir, "MC Dropout Accuracies by Prediction")
    data_plot.average_uncertainty_by_class(correct_sr, incorrect_sr, root_dir, "Softmax Response Accuracies by Prediction")
    data_plot.average_uncertainty_by_class(correct_BBB, incorrect_BBB, root_dir, "BbB Accuracies by Prediction")


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

    confusion_matrix = helper.make_confusion_matrix(predictions_BBB, test_data, test_indexes, True)
    data_plot.plot_confusion(confusion_matrix, root_dir, "BbB Response on Test Set with Cost matrix")
    confusion_matrix = helper.confusion_array(confusion_matrix, dimension=1)
    data_plot.plot_confusion(confusion_matrix, root_dir, "BbB Response Test Set Normalized with Cost matrix")

    confusion_matrix = helper.make_confusion_matrix(predictions_BBB, test_data, test_indexes, False)
    data_plot.plot_confusion(confusion_matrix, root_dir, "BbB Response on Test Set without Cost matrix")
    confusion_matrix = helper.confusion_array(confusion_matrix, dimension=1)
    data_plot.plot_confusion(confusion_matrix, root_dir, "BbB Response Test Set Normalized without Cost matrix")


for i in range(0, 1):
    
    network = model.Classifier(image_size, 8, class_weights, device, dropout=0.3, BBB=BBB)
    network.to(device)

    if BBB:

        # Set the learning rate to be higher for the Bayesian Layer
        BBB_weights = ['hidden_layer.weight_mu', 'hidden_layer.weight_rho', 'hidden_layer.bias_mu', 'hidden_layer.bias_rho',
                      'hidden_layer2.weight_mu', 'hidden_layer2.weight_rho', 'hidden_layer2.bias_mu', 'hidden_layer2.bias_rho']
        
        BBB_parameters = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in BBB_weights, network.named_parameters()))))
        base_parameters = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in BBB_weights, network.named_parameters()))))

        optim = optimizer.SGD([
                {'params': BBB_parameters},
                {'params': base_parameters, 'lr': 0.0001}
            ], lr=0.0001, momentum=0.9)
        
        scheduler = optimizer.lr_scheduler.CyclicLR(optim, base_lr=[0.0001, 0.0001], max_lr=[0.12, 0.03], step_size_up=(555 * 10), mode="exp_range", gamma=0.9999)

    else:
        optim = optimizer.SGD(network.parameters(), lr=0.0001, momentum=0.9)
        scheduler = optimizer.lr_scheduler.CyclicLR(optim, base_lr=0.0001, max_lr=0.03, step_size_up=(555 * 10), mode="exp_range", gamma=0.9999)

    
    ROOT_SAVE_DIR = SAVE_DIR
    
    if not os.path.exists(ROOT_SAVE_DIR):
        os.mkdir(ROOT_SAVE_DIR)
        
    
    if BBB:
        SAVE_DIR += f"/BBB_Classifier_{i}/"
    elif TRAIN_MC_DROPOUT:
        SAVE_DIR += f"/MC_Classifier_{i}/"
    else:
        SAVE_DIR += f"/SM_Classifier_{i}/"

    if not LOAD:
        starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = train(SAVE_DIR, 0,
                                                                                       [], [],
                                                                                       [], [],
                                                                                       verbose=True)

    if BBB:
        network, optim, scheduler, starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = helper.load_net(SAVE_DIR, 8, image_size, device, class_weights)

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
        predictions_BBB = helper.read_rows(SAVE_DIR + "BBB_entropy_predictions.csv")

    if SOFTMAX:

        network, optim, scheduler, starting_epoch, val_losses, train_losses, val_accuracies, train_accuracies = helper.load_net(SAVE_DIR, 8, image_size, device, class_weights)

        if not os.path.exists(SAVE_DIR + "entropy/"):
            os.mkdir(SAVE_DIR + "entropy/")
            os.mkdir(SAVE_DIR + "variance/")
            os.mkdir(SAVE_DIR + "costs/")

        if ISIC_pred:
            predictions_softmax, entropy_soft, costs_softmax = testing.predict(ISIC_set, SAVE_DIR, network, len(ISIC_data), device,
                                                                 softmax=True, ISIC=True)
            predictions_softmax.insert(0, ["image","MEL","NV","BCC","AK","BKL","DF","VASC","SCC","UNK"])
        else:
            predictions_softmax, entropy_soft, costs_softmax = testing.predict(test_set, SAVE_DIR, network, test_size, device,
                                                             softmax=True)
        helper.write_rows(predictions_softmax, SAVE_DIR + "softmax_predictions.csv")
        helper.write_rows(entropy_soft, SAVE_DIR + "softmax_entropy.csv")
        helper.write_rows(costs_softmax, SAVE_DIR + "softmax_costs.csv")

        predictions_softmax = helper.read_rows(SAVE_DIR + "softmax_predictions.csv")
        costs_sr = helper.read_rows(SAVE_DIR + "softmax_costs.csv")

        predictions_softmax = helper.string_to_float(predictions_softmax)
        costs_sr = helper.string_to_float(costs_sr)

        if ISIC_pred:
             predictions_mc_entropy, predictions_mc_var, costs_mc = testing.predict(ISIC_set, SAVE_DIR, network,
                                                                                   len(ISIC_data),
                                                                                   device, mc_dropout=True,
                                                                                   forward_passes=FORWARD_PASSES,
                                                                                   ISIC=True)
             predictions_mc_entropy.insert(0, ["image", "MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"])
        else:
            predictions_mc_entropy, predictions_mc_var, costs_mc = testing.predict(test_set, SAVE_DIR, network, test_size,
                                                                               device, mc_dropout=True,
                                                                               forward_passes=FORWARD_PASSES)
        helper.write_rows(predictions_mc_entropy, SAVE_DIR + "mc_entropy_predictions.csv")
        helper.write_rows(predictions_mc_var, SAVE_DIR + "mc_variance_predictions.csv")
        helper.write_rows(costs_mc, SAVE_DIR + "mc_costs.csv")

        predictions_mc = helper.read_rows(SAVE_DIR + "mc_entropy_predictions.csv")
        costs_mc = helper.read_rows(SAVE_DIR + "mc_costs.csv")

        predictions_mc = helper.string_to_float(predictions_mc)
        costs_mc = helper.string_to_float(costs_mc)


SAVE_DIR = "saved_models/SM_Classifier_0/"
predictions_mc = helper.read_rows(SAVE_DIR + "mc_entropy_predictions.csv")
costs_mc = helper.read_rows(SAVE_DIR + "mc_costs.csv")

SAVE_DIR = "saved_models/SM_Classifier_0/"

predictions_softmax = helper.read_rows(SAVE_DIR + "softmax_entropy.csv")
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

#predictions_BBB = list(helper.get_bayes_preds(predictions_BBB))
#predictions_softmax = list(helper.get_bayes_preds(predictions_softmax))
#predictions_mc = list(helper.get_bayes_preds(predictions_mc))

print_metrics(SAVE_DIR)


