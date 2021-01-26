"""
Helper.py: File responsible for random helpful functions, such as; loading and saving data, calculating
means and standard deviation etc.
"""

import torch
import model
from tqdm import tqdm
import csv
import torch.optim as optimizer
from copy import deepcopy
import numpy as np

LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC'}

def plot_image_at_index(data_plot, data_loader, index):

    data = data_loader[index]
    data_plot.show_data(data)
    print(index, data['image'].size(), LABELS[data['label']])
    # Only display the first X images


def plot_set(data_set, data_plot, stop, batch_stop):
    """
    displays image from data set
    :param data_set: data set to display images from
    :param data_plot: plotting class that uses matplotlib to display samples
    """

    for i_batch, sample_batch in enumerate(data_set):
        print(i_batch, sample_batch['image'].size(),
              sample_batch['label'].size())

        if i_batch == stop:
            data_plot.show_batch(sample_batch, batch_stop)
            break


def save_net(net, optim, PATH):
    states = {'network': net.state_dict(),
             'optimizer': optim.state_dict()}
    torch.save(states, PATH)

def change_to_device(network, optim, device):
    network = network.to(device)

    for state in optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    return network, optim

def load_net(PATH, image_size, output_size):
    net = model.Classifier(image_size, output_size)
    optim = optimizer.Adam(net.parameters(), lr=0.001)
    states = torch.load(PATH)

    optim.load_state_dict(states['optimizer'])
    net.load_state_dict(states['network'])
    net.train()
    return net, optim

def get_mean_and_std(data_set):
    """
    Cycles over data set and channels, calculating the mean and standard deviation
    for a RGB coloured data set
    :param data_set: Data set to calculate mean and std for
    :return: List of mean and standard deviations per channel
    """
    colour_sum = 0
    channel_squared = 0

    print("\nCalculating mean and std:\n")

    for i_batch, sample_batch in enumerate(tqdm(data_set)):
        colour_sum += torch.mean(sample_batch['image'], dim=[0,2,3])
        channel_squared += torch.mean(sample_batch['image']**2, dim=[0,2,3])

    mean = colour_sum / len(data_set)
    std = (channel_squared/len(data_set) - mean**2)**0.5

    print(f"\nMean: {mean}")
    print(f"\nStandard Deviation: {std}")

    return mean, std

def read_csv(filename):

    list_to_return = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            list_to_return += row

    new_list = []

    for item in list_to_return:
        new_list.append(float(item))

    return new_list

def write_csv(list_to_write, filename):
    """
    writes a list of items to a output file, used mainly for saving loss and accuracy values
    :param list_to_write: list to write to csv
    :param filename: location and name of csv file
    :return:
    """

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list_to_write)

def count_classes(dataset, batch_size):
    LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC'}
    labels_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}
    print("\nCounting labels")

    for i_batch, sample_batch in enumerate(tqdm(dataset)):
        labels_batch = sample_batch['label']
        for i in range(len(labels_batch)):
            label = labels_batch[i].item()
            label = LABELS[label]
            try:
                labels_count[label] += 1
            except Exception as e:
                continue

    print(labels_count)

    for label, count in labels_count.items():
        print(f"{label}: {round(count / (len(dataset) * batch_size) * 100, 3)}%")
    temp = labels_count.values()
    temp = list(temp)
    temp = sum(temp)
    print(f"Total number of samples in the set is: {temp}")

    return labels_count

def write_rows(list_to_write, filename):
    """
    writes a list of items to a output file, used mainly for saving loss and accuracy values
    :param list_to_write: list to write to csv
    :param filename: location and name of csv file
    :return:
    """

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(list_to_write)

def read_rows(filname):

    with open(filname, 'r', newline='') as f:
        reader = csv.reader(f)
        arrays = list(reader)

    return arrays


def float_to_string(arrays):

    for c in range(0, len(arrays)):
        arrays[c] = '{:.17f}'.format(arrays[c])

    return arrays

def string_to_float(arrays):
    for c in range(0, len(arrays)):
        for i in range(0, len(arrays[c])):
            arrays[c][i] = float(arrays[c][i])

    return arrays

def get_cost(answer, real_answer, num_classes=8):

    """cost_matrix = [[0, 10, 10, 10, 10, 10, 10, 0],
                  [150, 0, 30, 20, 0, 0, 20, 150],
                  [10, 10, 0, 0, 10, 10, 0, 10],
                  [10, 10, 0, 0, 10, 10, 0, 10],
                  [150, 0, 30, 20, 0, 0, 20, 150],
                  [150, 0, 30, 20, 0, 0, 20, 150],
                  [10, 10, 0, 0, 10, 10, 0, 10],
                  [0, 10, 10, 10, 10, 10, 10, 0]]"""

    """    cost_matrix = [[1, 10, 10, 10, 10, 10, 10, 1],
                   [150, 1, 30, 20, 1, 1, 20, 150],
                   [10, 10, 1, 1, 10, 10, 1, 10],
                   [10, 10, 1, 1, 10, 10, 1, 10],
                   [150, 1, 30, 20, 1, 1, 20, 150],
                   [150, 1, 30, 20, 1, 1, 20, 150],
                   [10, 10, 1, 1, 10, 10, 1, 10],
                   [1, 10, 10, 10, 10, 10, 10, 1]]"""

    """cost_matrix = [[0.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 0.1],
                   [15.1, 0.1, 3.1, 2.1, 0.1, 0.1, 2.1, 15.1],
                   [1.1, 1.1, 0.1, 0.1, 1.1, 1.1, 0.1, 1.1],
                   [1.1, 1.1, 0.1, 0.1, 1.1, 1.1, 0.1, 1.1],
                   [15.1, 0.1, 3.1, 2.1, 0.1, 0.1, 2.1, 15.1],
                   [15.1, 0.1, 3.1, 2.1, 0.1, 0.1, 2.1, 15.1],
                   [1.1, 1.1, 0.1, 0.1, 1.1, 1.1, 0.1, 1.1],
                   [0.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 0.1]]"""

    """cost_matrix = [[1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1],
                   [15.1, 1, 3.1, 2.1, 1, 1, 2.1, 15.1],
                   [1.1, 1.1, 1, 1, 1.1, 1.1, 1, 1.1],
                   [1.1, 1.1, 1, 1, 1.1, 1.1, 1, 1.1],
                   [15.1, 1, 3.1, 2.1, 1, 1, 2.1, 15.1],
                   [15.1, 1, 3.1, 2.1, 1, 1, 2.1, 15.1],
                   [1.1, 1.1, 1, 1, 1.1, 1.1, 1, 1.1],
                   [1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1]]"""

    cost_matrix = [[1, 10, 10, 10, 10, 10, 10, 1],
                   [150, 1, 30, 20, 1, 1, 20, 150],
                   [10, 10, 1, 1, 10, 10, 1, 10],
                   [10, 10, 1, 1, 10, 10, 1, 10],
                   [150, 1, 30, 20, 1, 1, 20, 150],
                   [150, 1, 30, 20, 1, 1, 20, 150],
                   [10, 10, 1, 1, 10, 10, 1, 10],
                   [1, 10, 10, 10, 10, 10, 10, 1]]

    cost = torch.tensor(cost_matrix[answer.item()][real_answer.item()])


    return cost




def get_correct_incorrect(predictions, data_loader, test_indexs, threshold=-1.0):

    predictions = deepcopy(predictions)

    correct = []
    incorrect = []
    uncertain = []
    wrong = right = total = 0

    for index in test_indexs:

        uncertainty = predictions[total].pop()
        answer = np.argmax(predictions[total])
        real_answer = data_loader.get_label(index)


        if uncertainty <= threshold:
            uncertain.append([answer, uncertainty])
        elif answer == real_answer:
            correct.append([answer, uncertainty])
            right += 1
        else:
            incorrect.append([answer, uncertainty])
            wrong += 1
        total += 1

    return correct, incorrect, uncertain

def confusion_array(arrays):

    new_arrays = []

    for array in arrays:
        new_array = []

        for item in array:
            new_array.append(round(item / (sum(array) + 1), 4))

        new_arrays.append(new_array)

    return new_arrays


def make_confusion_matrix(predictions, data_loader, test_indexes, threshold=-1.0):
    predictions = deepcopy(predictions)

    confusion_matrix = []

    for i in range(8):
        confusion_matrix.append([0, 0, 0, 0, 0, 0, 0, 0])

    total = 0

    for index in test_indexes:

        predictions[total].pop()
        answer = np.argmax(predictions[total])
        real_answer = data_loader.get_label(index)

        confusion_matrix[real_answer][answer] += 1

        total += 1

    return confusion_matrix