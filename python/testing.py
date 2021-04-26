"""
testing.py: The file responsible for holding the methods relating to testing network on datasets. Does predictions
for the ISIC2019 Challenge and also predictions on already known images
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import helper

def softmax_pred(data_set, network, n_classes, n_samples, device, ISIC):

    costs = []
    entropies = []
    filenames = []
    predictions = np.empty((0, n_classes))
    soft_max = nn.Softmax(dim=1)

    network.eval()

    for i_batch, sample_batch in enumerate(tqdm(data_set)):
        image_batch = sample_batch['image'].to(device)
        filename_batch = sample_batch['filename']
        with torch.no_grad():
            outputs = soft_max(network(image_batch, dropout=False))

        for output in outputs:
            predictions = np.vstack((predictions, output.cpu().numpy()))

        for filename in filename_batch:
            filenames.append(filename)

    predictions_e = np.copy(predictions)
    predictions = predictions.tolist()
    predictions_e = predictions_e.tolist()

    for pred in predictions:
        pred.append(1 - max(pred))
        if np.max(pred) > 0.9999:
            entropies.append(0)
        else:
            entropies.append(-np.sum(pred * np.log2(pred), axis=0))

    for c in range(0, len(entropies)):
        entropies[c] = (entropies[c] - min(entropies)) / (max(entropies) - min(entropies))

    for i in range(0, len(predictions_e)):
        predictions_e[i].append(entropies[i])

    for i in range(0, len(predictions)):
        current_costs = helper.get_each_cost(predictions_e[i], uncertain=True)
        for c in range(0, len(current_costs)):
            current_costs[c] = '{:.17f}'.format(current_costs[c])
        costs.append(current_costs)


    #i = 0
    for i in range(0, len(predictions)):
        for c in range(0, len(predictions[i])):
            predictions[i][c] = '{:.17f}'.format(predictions[i][c])
            predictions_e[i][c] = '{:.17f}'.format(predictions_e[i][c])
        if ISIC:
            predictions[i].insert(0, filenames[i][:-4])


        #i = i + 1

    return predictions, predictions_e, costs


def monte_carlo(data_set, forward_passes, network, n_samples, n_classes, root_dir, device, BBB, ISIC):

    #costs = np.empty((0, n_samples, n_classes))
    # Add one for the entropy
    n_classes = n_classes + 1
    filenames =[]
    soft_max = nn.Softmax(dim=1)
    drop_predictions = np.empty((0, n_samples, n_classes))
    costs = np.empty((0, n_samples, n_classes))
    efficient_net_outputs = []

    network.eval()

    for i_batch, sample_batch in enumerate(tqdm(data_set)):
        image_batch = sample_batch['image'].to(device)
        filename_batch = sample_batch['filename']

        for filename in filename_batch:
            filenames.append(filename)

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
                    entropy = -np.sum(answers * np.log2(answers), axis=0)  # shape (n_samples, n_classes)

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
            helper.write_rows(mean_entropy, root_dir + f"entropy/BBB_forward_pass_{i}_entropy.csv")
            helper.write_rows(mean_variance, root_dir + f"variance/BBB_forward_pass_{i}_variance.csv")
            helper.write_rows(costs_mean, root_dir + f"costs/BBB_forward_pass_{i}_costs.csv")
        else:
            helper.write_rows(mean_entropy, root_dir + f"entropy/mc_forward_pass_{i}_entropy.csv")
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

    for i in range(0, len(mean_entropy)):

        for c in range(0, len(mean_entropy[i])):
            mean_entropy[i][c] = '{:.17f}'.format(mean_entropy[i][c])
        if ISIC:
            mean_entropy[i].insert(0, filenames[i][:-4])

    for preds in mean_variance:

        for c in range(0, len(preds)):
            preds[c] = '{:.17f}'.format(preds[c])

    for cost in costs_mean:

        for c in range(0, len(cost)):
            cost[c] = '{:.17f}'.format(cost[c])



    return mean_entropy, mean_variance, costs_mean


def predict(test_set, root_dir, network, num_samples, device, n_classes=8, mc_dropout=False, BBB=False, forward_passes=100, softmax=False, ISIC=False):

    print("Predicting on Test set")

    # Make sure network is in eval mode
    network.eval()

    if mc_dropout:
        predictions_e, predictions_v, costs = monte_carlo(test_set, forward_passes, network, num_samples, n_classes, root_dir, device, BBB, ISIC)
        return predictions_e, predictions_v, costs
    elif softmax:
        predictions, predictions_e, costs = softmax_pred(test_set, network, n_classes, num_samples, device, ISIC)
        return predictions, predictions_e, costs
    elif BBB:
        predictions_e, predictions_v, costs = monte_carlo(test_set, forward_passes, network, num_samples, n_classes,
                                                          root_dir, device, BBB, ISIC)
        return predictions_e, predictions_v, costs