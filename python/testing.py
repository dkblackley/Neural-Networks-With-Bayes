"""
testing.py: The file responsible for holding the methods relating to testing network on datasets. Does predictions
for the ISIC2019 Challenge and also predictions on already known images
"""

import torch
import torch.nn as nn
import sys
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
import helper

def softmax_pred_ISIC(outputs, data_loader, i_batch, batch_size):

    predictions = []

    for i in range(0, batch_size):

        try:
            answers = outputs[i].tolist()

            new_list = []
            for answer in answers:
                new_list.append(answer)

            new_list.append(1.0 - max(new_list))

            for c in range(0, len(new_list)):
                new_list[c] = '{:.17f}'.format(new_list[c])

            answers = new_list

            answers.insert(0, data_loader.get_filename(i + (batch_size * i_batch))[:-4])
            predictions.append(answers)
        # for the last batch, which won't be perfectly of size BATCH_SIZE
        except Exception as e:
            break

    return predictions

def monte_carlo_ISIC(data_set, data_loader, forward_passes, network):
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
                    outputs = soft_max(network(image_batch, dropout=True, drop_rate=0.25))

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


def predict_ISIC(data_set, data_loader, network, device, forward_passes, softmax=False, mc_dropout=False):
    """
    Predicts on a data set without labels, returns a list of lists in a form suitable for the ISIC2019 Leaderboards
    :param data_set: data set to predict on
    :param data_loader: data loader to get the filename from
    :return: a list of lists holding the networks predictions for each class
    """

    print("\nPredicting on ISIC 2019 Test set")

    if softmax:
        predictions = [['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']]
        for i_batch, sample_batch in enumerate(tqdm(data_set)):

            image_batch = sample_batch['image']

            image_batch = image_batch.to(device)
            soft_max = nn.Softmax(dim=1)
            with torch.no_grad():
                outputs = soft_max(network(image_batch, dropout=False))
            predictions.extend(softmax_pred_ISIC(outputs, data_loader, i_batch))

    elif mc_dropout:
        predictions = monte_carlo_ISIC(data_set, data_loader, forward_passes)
        predictions.insert(0, ['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'])

    if softmax:
        helper.write_rows(predictions, "saved_model/softmax_predictions.csv")
    elif mc_dropout:
        helper.write_rows(predictions, "saved_model/dropout_predictions.csv")

    return predictions

def softmax_pred(data_set, network, n_classes, n_samples, device):

    costs = []
    predictions = np.empty((0, n_classes))
    soft_max = nn.Softmax(dim=1)

    network.eval()

    for i_batch, sample_batch in enumerate(tqdm(data_set)):
        image_batch = sample_batch['image'].to(device)
        with torch.no_grad():
            outputs = soft_max(network(image_batch, dropout=False))

        for output in outputs:
            predictions = np.vstack((predictions, output.cpu().numpy()))

    predictions = predictions.tolist()

    for pred in predictions:
        pred.append(1 - max(pred))

    for preds in predictions:
        current_costs = helper.get_each_cost(preds, uncertain=True)
        for c in range(0, len(current_costs)):
            current_costs[c] = '{:.17f}'.format(current_costs[c])
        costs.append(current_costs)


    #i = 0
    for preds in predictions:

        #preds.append(entropies[i])

        for c in range(0, len(preds)):
            preds[c] = '{:.17f}'.format(preds[c])

        #i = i + 1

    return predictions, costs


def monte_carlo(data_set, forward_passes, network, n_samples, n_classes, root_dir, device, BBB):

    #costs = np.empty((0, n_samples, n_classes))
    # Add one for the entropy
    n_classes = n_classes + 1
    soft_max = nn.Softmax(dim=1)
    drop_predictions = np.empty((0, n_samples, n_classes))
    costs = np.empty((0, n_samples, n_classes))
    efficient_net_outputs = []

    network.eval()

    for i_batch, sample_batch in enumerate(data_set):
        image_batch = sample_batch['image'].to(device)
        with torch.no_grad():
            efficient_net_output = network.extract_efficientNet(image_batch)
        efficient_net_outputs.append(efficient_net_output)

    for i in tqdm(range(0, forward_passes)):

        predictions = np.empty((0, n_classes))
        current_costs = np.empty((0, n_classes))


        for c in range(0, len(efficient_net_outputs)):
            #image_batch = sample_batch['image'].to(device)

            with torch.no_grad():
                if BBB:
                    outputs = soft_max(network.pass_through_layers(efficient_net_outputs[c]))
                else:
                    outputs = soft_max(network.pass_through_layers(efficient_net_outputs[c], dropout=True))
            for output in outputs:
                answers = output.cpu().numpy()

                # avoid log(0) errors
                if np.max(answers) > 0.9999:
                    entropy = 0.0
                else:
                    entropy = -np.sum(answers * np.log(answers), axis=0)  # shape (n_samples, n_classes)

                #current_costs = np.vstack((current_costs, helper.get_each_cost(answers)))
                answers = np.append(answers, entropy)
                current_costs = np.vstack((current_costs, helper.get_each_cost(answers, uncertain=True)))
                predictions = np.vstack((predictions, answers))

        drop_predictions = np.vstack((drop_predictions, predictions[np.newaxis, :, :]))
        temp = np.delete(drop_predictions, n_classes - 1, 2) # Remove entropy

        costs = np.vstack((costs, current_costs[np.newaxis, :, :]))
        costs_mean = np.mean(costs, axis=0)

        mean_entropy = mean_variance = np.mean(drop_predictions, axis=0)
        variance = np.var(temp, axis=0, ddof=1)  # shape (n_samples, n_classes)

        variance = variance.tolist()
        mean_entropy = mean_entropy.tolist()
        mean_variance = mean_variance.tolist()
        costs_mean = costs_mean.tolist()
        entropies = [c[n_classes - 1] for c in mean_entropy]

        for c in range(0, len(entropies)):
            entropies[c] = (entropies[c] - min(entropies)) / (max(entropies) - min(entropies))

        for c in range(0, len(mean_entropy)):
            mean_entropy[c][n_classes - 1] = entropies[c]
            mean_variance[c][n_classes - 1] = sum(variance[c])

        for preds in mean_entropy:

            for c in range(0, len(preds)):
                preds[c] = '{:.17f}'.format(preds[c])

        for preds in mean_variance:

            for c in range(0, len(preds)):
                preds[c] = '{:.17f}'.format(preds[c])

        for cost in costs_mean:
            for c in range(0, len(cost)):
                cost[c] = '{:.17f}'.format(cost[c])

        if i == 3:
            temp2 = np.array(mean_variance)
        if BBB:
            helper.write_rows(mean_entropy, root_dir + f"naturallog/BBB_forward_pass_{i}_entropy.csv")
            helper.write_rows(mean_variance, root_dir + f"variance/BBB_forward_pass_{i}_variance.csv")
            helper.write_rows(costs_mean, root_dir + f"costs/BBB_forward_pass_{i}_costs.csv")
        else:
            helper.write_rows(mean_entropy, root_dir + f"naturallog/mc_forward_pass_{i}_entropy.csv")
            helper.write_rows(mean_variance, root_dir + f"variance/mc_forward_pass_{i}_variance.csv")
            helper.write_rows(costs_mean, root_dir + f"costs/mc_forward_pass_{i}_costs.csv")




    mean_entropy = np.mean(drop_predictions, axis=0)  # shape (n_samples, n_classes)
    mean_variance = np.mean(drop_predictions, axis=0)  # shape (n_samples, n_classes)
    costs_mean = np.mean(costs, axis=0)
    temp = np.delete(drop_predictions, n_classes - 1, 2)
    variance = np.var(temp, axis=0)  # shape (n_samples, n_classes)

    variance = variance.tolist()
    mean_entropy = mean_entropy.tolist()
    mean_variance = mean_variance.tolist()
    entropies = [c[n_classes - 1] for c in mean_entropy]

    for c in range(0, len(entropies)):
        entropies[c] = (entropies[c] - min(entropies)) / (max(entropies) - min(entropies))

    for c in range(0, len(mean_entropy)):
        mean_entropy[c][n_classes - 1] = entropies[c]
        mean_variance[c][n_classes - 1] = sum(variance[c])

    for preds in mean_entropy:

        for c in range(0, len(preds)):
            preds[c] = '{:.17f}'.format(preds[c])

    for preds in mean_variance:

        for c in range(0, len(preds)):
            preds[c] = '{:.17f}'.format(preds[c])

    for cost in costs_mean:

        for c in range(0, len(cost)):
            cost[c] = '{:.17f}'.format(cost[c])

    return mean_entropy, mean_variance, costs_mean


def predict(test_set, root_dir, network, num_samples, device, n_classes=8, mc_dropout=False, BBB=False, forward_passes=100, softmax=False):

    print("Predicting on Test set")

    # Make sure network is in eval mode
    network.eval()

    if mc_dropout:
        predictions_e, predictions_v, costs = monte_carlo(test_set, forward_passes, network, num_samples, n_classes, root_dir, device, BBB)
        return predictions_e, predictions_v, costs
    elif softmax:
        predictions, costs = softmax_pred(test_set, network, n_classes, num_samples, device)
        return predictions, costs
    elif BBB:
        predictions_e, predictions_v, costs = monte_carlo(test_set, forward_passes, network, num_samples, n_classes,
                                                          root_dir, device, BBB)
        return predictions_e, predictions_v, costs