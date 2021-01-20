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


def predict_ISIC(data_set, data_loader, network, device, forward_passes, softmax=False, mc_dropout=False):
    """
    Predicts on a data set without labels, returns a list of lists in a form suitable for the ISIC2019 Leaderboards
    :param data_set: data set to predict on
    :param data_loader: data loader to get the filename from
    :return: a list of lists holding the networks predictions for each class
    """

    print("\nPredicting on ISIC2019 Test set")

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

def softmax_pred(data_set, network, n_classes):

    predictions = np.empty((0, n_classes))
    soft_max = nn.Softmax(dim=1)

    for i_batch, sample_batch in enumerate(tqdm(data_set)):
        image_batch = sample_batch['image']
        with torch.no_grad():
            outputs = soft_max(network(image_batch, dropout=True))

        for output in outputs:
            predictions = np.vstack((predictions, output.cpu().numpy()))

    epsilon = sys.float_info.min
    entropies = -np.sum(predictions*np.log10(predictions + epsilon), axis=-1)

    predictions = predictions.tolist()
    entropies = entropies.tolist()

    for i in range(0, len(entropies)):
        entropies[i] = (entropies[i] - min(entropies)) / (max(entropies) - min(entropies))

    i = 0
    for preds in predictions:

        preds.append(entropies[i])

        for c in range(0, len(preds)):
            preds[c] = '{:.17f}'.format(preds[c])

        i = i + 1

    return predictions


def monte_carlo(data_set, forward_passes, network, n_samples, n_classes):

    soft_max = nn.Softmax(dim=1)
    drop_predictions = np.empty((0, n_samples, n_classes))

    for i in range(0, forward_passes):
        print(f"\n\n Forward pass {i + 1} of {forward_passes}\n")
        predictions = np.empty((0, n_classes))


        for i_batch, sample_batch in enumerate(tqdm((data_set), position=0, leave=True)):
            image_batch = sample_batch['image']

            with torch.no_grad():
                outputs = soft_max(network(image_batch, dropout=True))

            for output in outputs:
                predictions = np.vstack((predictions, output.cpu().numpy()))

        drop_predictions = np.vstack((drop_predictions, predictions[np.newaxis, :, :]))

    mean = np.mean(drop_predictions, axis=0)  # shape (n_samples, n_classes)

    epsilon = sys.float_info.min
    entropies = -np.sum(mean*np.log10(mean + epsilon), axis=-1)  # shape (n_samples, n_classes)

    mean = mean.tolist()
    entropies = entropies.tolist()

    for i in range(0, len(entropies)):
        entropies[i] = (entropies[i] - min(entropies))/(max(entropies) - min(entropies))

    i = 0
    for preds in mean:

        preds.append(entropies[i])

        for c in range(0, len(preds)):
            preds[c] = '{:.17f}'.format(preds[c])

        i = i + 1

    return mean


def predict(test_set, network, num_samples, n_classes=8, mc_dropout=False, forward_passes=100, softmax=False):

    print("Predicting on Test set")

    # Make sure network is in eval mode
    network.eval()

    predictions = []

    if mc_dropout:
        predictions = monte_carlo(test_set, forward_passes, network, num_samples, n_classes)
    elif softmax:
        predictions = softmax_pred(test_set, network, n_classes)

    return predictions