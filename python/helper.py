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
import os


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


def save_net(net, optim, lr_sched, PATH):
    states = {'network': net.state_dict(),
             'optimizer': optim.state_dict(),
             'lr_sched': lr_sched.state_dict()}
    torch.save(states, PATH)

def change_to_device(network, optim, device):
    network = network.to(device)

    for state in optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    return network, optim

def read_net(PATH, image_size, output_size, device, class_weights):
    net = model.Classifier(image_size, output_size, device, class_weights)
    #optim = optimizer.Adam(net.parameters(), lr=0.001)
    optim = optimizer.SGD(net.parameters(), lr=0.00001)
    scheduler = optimizer.lr_scheduler.CyclicLR(optim, base_lr=0.0001, max_lr=0.03, step_size_up=(555 * 10))
    states = torch.load(PATH, map_location=device)

    try:
        net.load_state_dict(states['network'])
        optim.load_state_dict(states['optimizer'])
        scheduler.load_state_dict(states['lr_sched'])
    except Exception as e:
        net = model.Classifier(image_size, output_size, device, class_weights, BBB=True)
        
        BBB_weights = ['hidden_layer.weight_mu', 'hidden_layer.weight_rho', 'hidden_layer.bias_mu', 'hidden_layer.bias_rho',
                      'hidden_layer2.weight_mu', 'hidden_layer2.weight_rho', 'hidden_layer2.bias_mu', 'hidden_layer2.bias_rho']
        BBB_parameters = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in BBB_weights, net.named_parameters()))))
        base_parameters = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in BBB_weights, net.named_parameters()))))
        
    
        optim = optimizer.SGD([
                {'params': BBB_parameters},
                {'params': base_parameters, 'lr': 0.0001}
            ], lr=0.0001, momentum=0.9)
        scheduler = optimizer.lr_scheduler.CyclicLR(optim, base_lr=[0.0001, 0.0001], max_lr=[0.12, 0.03], step_size_up=(555 * 10), mode="exp_range", gamma=0.9999)

        net.load_state_dict(states['network'])
        optim.load_state_dict(states['optimizer']) 
        scheduler.load_state_dict(states['lr_sched'])

    net.train()
    
    net, optim = change_to_device(net, optim, device)
    
    return net, optim, scheduler

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

def attach_last_row(array1, array2):

    for i in range(0, len(array2)):
        temp = array2[i][-1]
        array1[i].append(array2[i][-1])

    return array1

def read_rows(filname):

    with open(filname, 'r', newline='') as f:
        reader = csv.reader(f)
        arrays = list(reader)

    return arrays

def calculate_sens_spec_acc(predictions):
    TP = 0
    FN = 0
    CP = 0
    CN = 0


def float_to_string(arrays):

    for c in range(0, len(arrays)):
        arrays[c] = '{:.17f}'.format(arrays[c])

    return arrays

def string_to_float(arrays):
    for c in range(0, len(arrays)):
        for i in range(0, len(arrays[c])):
            arrays[c][i] = float(arrays[c][i])

    return arrays

def normalize_matrix(matrix, min_val=0, max_val=150):

    for array in matrix:
        for i in range(0, len(array)):

            if array[i] != 1:
                array[i] = 1 + ((array[i] - min_val) / (max_val - min_val))
            else:
                array[i] = float(array[i])




    return matrix


def get_cost_matrix():
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
                   [15.1, 1, 3.1, 2.1, 1, 1, 2.1, 16.5],
                   [1.1, 1.1, 1, 1, 1.1, 1.1, 1, 1.1],
                   [1.1, 1.1, 1, 1, 1.1, 1.1, 1, 1.1],
                   [16.5, 1, 3.3, 2.2, 1, 1, 2.2, 16.5],
                   [16.5, 1, 3.3, 2.2, 1, 1, 2.2, 16.5],
                   [1.1, 1.1, 1, 1, 1.1, 1.1, 1, 1.1],
                   [1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1]]"""

    """    cost_matrix = [[1, 10, 10, 10, 10, 10, 10, 1],
                   [150, 1, 30, 20, 1, 1, 20, 150],
                   [10, 10, 1, 1, 10, 10, 1, 10],
                   [10, 10, 1, 1, 10, 10, 1, 10],
                   [150, 1, 30, 20, 1, 1, 20, 150],
                   [150, 1, 30, 20, 1, 1, 20, 150],
                   [10, 10, 1, 1, 10, 10, 1, 10],
                   [1, 10, 10, 10, 10, 10, 10, 1]]"""
    # LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC'}

    """cost_matrix = [[0, 150, 10, 10, 150, 150, 10, 0],
                   [10, 0, 10, 10, 0, 0, 10, 10],
                   [10, 30, 0, 0, 30, 30, 0, 10],
                   [10, 20, 0, 0, 20, 20, 0, 10],
                   [10, 0, 10, 10, 0, 0, 10, 10],
                   [10, 0, 10, 10, 0, 0, 10, 10],
                   [10, 20, 0, 0, 20, 20, 0, 10],
                   [0, 150, 10, 10, 150, 150, 10, 0]]"""

    cost_matrix = [[1, 150, 10, 10, 150, 150, 10, 1],
                   [10, 1, 10, 10, 1, 1, 10, 10],
                   [10, 30, 1, 1, 30, 30, 1, 10],
                   [10, 20, 1, 1, 20, 20, 1, 10],
                   [10, 1, 10, 10, 1, 1, 10, 10],
                   [10, 1, 10, 10, 1, 1, 10, 10],
                   [10, 20, 1, 1, 20, 20, 1, 10],
                   [1, 150, 10, 10, 150, 150, 10, 1]]

    normalize_matrix(cost_matrix)

    """cost_matrix = [[1, 16.5, 1.1, 1.1, 16.5, 16.5, 1.1, 1],
                   [1.1, 1, 1.1, 1.1, 1, 1, 1.1, 1.1],
                   [1.1, 3.3, 1, 1, 3.3, 3.3, 1, 1.1],
                   [1.1, 2.2, 1, 1, 2.2, 2.2, 1, 1.1],
                   [1.1, 1, 1.1, 1.1, 1, 1, 1.1, 1.1],
                   [1.1, 1, 1.1, 1.1, 1, 1, 1.1, 1.1],
                   [1.1, 2.2, 1, 1, 2.2, 2.2, 1, 1.1],
                   [1, 16.5, 1.1, 1.1, 16.5, 16.5, 1.1, 1]]"""



    return cost_matrix

def find_best_answer():
    pass

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
                   [15.1, 1, 3.1, 2.1, 1, 1, 2.1, 16.5],
                   [1.1, 1.1, 1, 1, 1.1, 1.1, 1, 1.1],
                   [1.1, 1.1, 1, 1, 1.1, 1.1, 1, 1.1],
                   [16.5, 1, 3.3, 2.2, 1, 1, 2.2, 16.5],
                   [16.5, 1, 3.3, 2.2, 1, 1, 2.2, 16.5],
                   [1.1, 1.1, 1, 1, 1.1, 1.1, 1, 1.1],
                   [1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1]]"""

    """    cost_matrix = [[1, 10, 10, 10, 10, 10, 10, 1],
                   [150, 1, 30, 20, 1, 1, 20, 150],
                   [10, 10, 1, 1, 10, 10, 1, 10],
                   [10, 10, 1, 1, 10, 10, 1, 10],
                   [150, 1, 30, 20, 1, 1, 20, 150],
                   [150, 1, 30, 20, 1, 1, 20, 150],
                   [10, 10, 1, 1, 10, 10, 1, 10],
                   [1, 10, 10, 10, 10, 10, 10, 1]]"""
    #LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC'}


    """cost_matrix = [[0, 150, 10, 10, 150, 150, 10, 0],
                   [10, 0, 10, 10, 0, 0, 10, 10],
                   [10, 30, 0, 0, 30, 30, 0, 10],
                   [10, 20, 0, 0, 20, 20, 0, 10],
                   [10, 0, 10, 10, 0, 0, 10, 10],
                   [10, 0, 10, 10, 0, 0, 10, 10],
                   [10, 20, 0, 0, 20, 20, 0, 10],
                   [0, 150, 10, 10, 150, 150, 10, 0]]"""

    cost_matrix = [[1, 150, 10, 10, 150, 150, 10, 1],
                   [10, 1, 10, 10, 1, 1, 10, 10],
                   [10, 30, 1, 1, 30, 30, 1, 10],
                   [10, 20, 1, 1, 20, 20, 1, 10],
                   [10, 1, 10, 10, 1, 1, 10, 10],
                   [10, 1, 10, 10, 1, 1, 10, 10],
                   [10, 20, 1, 1, 20, 20, 1, 10],
                   [1, 150, 10, 10, 150, 150, 10, 1]]

    normalize_matrix(cost_matrix)

    """cost_matrix = [[1, 16.5, 1.1, 1.1, 16.5, 16.5, 1.1, 1],
                   [1.1, 1, 1.1, 1.1, 1, 1, 1.1, 1.1],
                   [1.1, 3.3, 1, 1, 3.3, 3.3, 1, 1.1],
                   [1.1, 2.2, 1, 1, 2.2, 2.2, 1, 1.1],
                   [1.1, 1, 1.1, 1.1, 1, 1, 1.1, 1.1],
                   [1.1, 1, 1.1, 1.1, 1, 1, 1.1, 1.1],
                   [1.1, 2.2, 1, 1, 2.2, 2.2, 1, 1.1],
                   [1, 16.5, 1.1, 1.1, 16.5, 16.5, 1.1, 1]]"""




    #cost = torch.tensor(cost_matrix[answer.item()][real_answer.item()])
    cost = torch.tensor(cost_matrix[real_answer.item()][answer.item()])


    return cost


def find_lowest_cost(probabilities, num_classes=8, uncertain=False):

    if uncertain:
        cost_matrix = np.array([
                    [0, 150, 10, 10, 150, 150, 10, 1, 10],
                    [10, 0, 10, 10, 1, 1, 10, 10, 10],
                    [10, 30, 0, 1, 30, 30, 1, 10, 10],
                    [10, 20, 1, 0, 20, 20, 1, 10, 10],
                    [10, 1, 10, 10, 0, 1, 10, 10, 10],
                    [10, 1, 10, 10, 1, 0, 10, 10, 10],
                    [10, 20, 1, 1, 20, 20, 0, 10, 10],
                    [1, 150, 10, 10, 150, 150, 10, 0, 10],
                    [10, 10, 10, 10, 10, 10, 10, 10, 0]])
    else:
        cost_matrix = np.array([
                   [0, 150, 10, 10, 150, 150, 10, 1],
                   [10, 0, 10, 10, 1, 1, 10, 10],
                   [10, 30, 0, 1, 30, 30, 1, 10],
                   [10, 20, 1, 0, 20, 20, 1, 10],
                   [10, 1, 10, 10, 0, 1, 10, 10],
                   [10, 1, 10, 10, 1, 0, 10, 10],
                   [10, 20, 1, 1, 20, 20, 0, 10],
                   [1, 150, 10, 10, 150, 150, 10, 0]])

    """cost_matrix = np.array([
                   [0, 150, 10, 10],
                   [10, 0, 10, 10],
                   [10, 30, 0, 1],
                   [10, 20, 1, 0]])"""

    lowest_cost = -1
    answer = 0

    for j in range(len(probabilities)):
        total = 0

        for k in range(0, len(probabilities)):
            temp = cost_matrix[k][j]
            total += probabilities[k] * cost_matrix[k][j]

        if lowest_cost > total or lowest_cost == -1:
            answer = j
            lowest_cost = total

    #if np.argmax(probabilities) == 6:
    #    print('oi')

    return answer, lowest_cost

def get_label_indexes(predictions, test_indexes, data_loader):
    indexes = {'MEL': [], 'NV': [], 'BCC': [], 'AK': [], 'BKL': [], 'DF': [], 'VASC': [], 'SCC': []}
    new_predictions = {'MEL': [], 'NV': [], 'BCC': [], 'AK': [], 'BKL': [], 'DF': [], 'VASC': [], 'SCC': []}

    for i in range(0, len(test_indexes)):

        label = data_loader.get_label(test_indexes[i])
        indexes[LABELS[label]].append(test_indexes[i])
        new_predictions[LABELS[label]].append(predictions[i])

    return new_predictions, indexes


def find_true_cost(prediction, answer, num_classes=8, uncertain=False, flatten=False):

    if flatten:
        cost_matrix = [
            [0, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 0]]
    else:
        cost_matrix = [
                   [0, 150, 10, 10, 150, 150, 10, 1],
                   [10, 0, 10, 10, 1, 1, 10, 10],
                   [10, 30, 0, 1, 30, 30, 1, 10],
                   [10, 20, 1, 0, 20, 20, 1, 10],
                   [10, 1, 10, 10, 0, 1, 10, 10],
                   [10, 1, 10, 10, 1, 0, 10, 10],
                   [10, 20, 1, 1, 20, 20, 0, 10],
                   [1, 150, 10, 10, 150, 150, 10, 0]]


    if uncertain:
        if not flatten:
            for row in cost_matrix:
                row.append(10)
            cost_matrix.append([10, 10, 10, 10, 10, 10, 10, 10, 0])
        else:
            for row in cost_matrix:
                row.append(1)
            cost_matrix.append([1, 1, 1, 1, 1, 1, 1, 1, 0])

    return cost_matrix[answer][prediction]

def remove_values_below_threshold(list, threshold):
    pass

def remove_last_row(arrays):
    for array in arrays:
        array.pop()

def test_lowest_cost(probabilities, num_classes=8):
    cost_matrix = np.array([[0, 2],
                            [1, 0]])

    lowest_cost = -1
    answer = 0

    for j in range(len(probabilities)):
        total = 0

        for k in range(0, len(probabilities)):
            total += probabilities[k] * cost_matrix[k][j]

        if lowest_cost > total or lowest_cost == -1:
            answer = j
            lowest_cost = total

    return answer, lowest_cost

def get_each_cost(probabilities, uncertain=False):

    costs = []

    if uncertain:
        cost_matrix = np.array([
            [0, 150, 10, 10, 150, 150, 10, 1, 10],
            [10, 0, 10, 10, 1, 1, 10, 10, 10],
            [10, 30, 0, 1, 30, 30, 1, 10, 10],
            [10, 20, 1, 0, 20, 20, 1, 10, 10],
            [10, 1, 10, 10, 0, 1, 10, 10, 10],
            [10, 1, 10, 10, 1, 0, 10, 10, 10],
            [10, 20, 1, 1, 20, 20, 0, 10, 10],
            [1, 150, 10, 10, 150, 150, 10, 0, 10],
            [10, 10, 10, 10, 10, 10, 10, 10, 0]])
    else:
        cost_matrix = np.array([
            [0, 150, 10, 10, 150, 150, 10, 1],
            [10, 0, 10, 10, 1, 1, 10, 10],
            [10, 30, 0, 1, 30, 30, 1, 10],
            [10, 20, 1, 0, 20, 20, 1, 10],
            [10, 1, 10, 10, 0, 1, 10, 10],
            [10, 1, 10, 10, 1, 0, 10, 10],
            [10, 20, 1, 1, 20, 20, 0, 10],
            [1, 150, 10, 10, 150, 150, 10, 0]])

    for j in range(0, len(probabilities)):
        total = 0

        for k in range(0, len(probabilities)):
            temp = cost_matrix[k][j]
            total += probabilities[k] * cost_matrix[k][j]

        costs.append(total)


    return costs



def get_correct_incorrect(predictions, data_loader, test_indexes, cost_matrix, threshold=-1.0):

    predictions = deepcopy(predictions)

    correct = []
    incorrect = []
    uncertain = []
    wrong = right = total = 0

    for index in test_indexes:

        uncertainty = predictions[total].pop()
        if cost_matrix:
            answer = find_lowest_cost(predictions[total])[0]
        else:
            answer = np.argmax(predictions[total])
        real_answer = data_loader.get_label(index)


        if uncertainty <= threshold:
            uncertain.append([answer, real_answer, uncertainty])
        elif answer == real_answer:
            correct.append([answer, real_answer, uncertainty])
            right += 1
        else:
            incorrect.append([answer, real_answer, uncertainty])
            wrong += 1
        total += 1

    return correct, incorrect, uncertain

def is_prediction_corect(prediction, index, data_loader):

    real_answer = data_loader.get_label(index)

    if prediction == real_answer:
        return True
    else:
        return False

def confusion_array(arrays, dimension=0):

    np_array = np.array(arrays)

    summed_array = np_array.sum(axis=dimension, keepdims=1)

    new_arrays = np.divide(np_array, summed_array, out=np.zeros(np_array.shape, dtype=float),
                                                  where=summed_array!=0)
    new_arrays = np.round(new_arrays, 3)

    return new_arrays.tolist()


def make_confusion_matrix(predictions, data_loader, test_indexes, cost_matrix, threshold=-1.0):
    predictions = deepcopy(predictions)

    confusion_matrix = []

    for i in range(8):
        confusion_matrix.append([0, 0, 0, 0, 0, 0, 0, 0])

    total = 0

    for index in test_indexes:

        predictions[total].pop()
        if cost_matrix:
            answer = find_lowest_cost(predictions[total])[0]
        else:
            answer = np.argmax(predictions[total])
        real_answer = data_loader.get_label(index)

        confusion_matrix[real_answer][answer] += 1

        total += 1

    return confusion_matrix


def save_network(network, optim, scheduler, val_losses, train_losses, val_accuracies, train_accuracies, root_dir):
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)

    save_net(network, optim, scheduler, root_dir + "model_parameters")
    write_csv(val_losses, root_dir + "val_losses.csv")
    write_csv(train_losses, root_dir + "train_losses.csv")
    write_csv(val_accuracies, root_dir + "val_accuracies.csv")
    write_csv(train_accuracies, root_dir + "train_accuracies.csv")


def load_net(root_dir, output_size, image_size, device, class_weights):
    val_losses = read_csv(root_dir + "val_losses.csv")
    train_losses = read_csv(root_dir + "train_losses.csv")
    val_accuracies = read_csv(root_dir + "val_accuracies.csv")
    train_accuracies = read_csv(root_dir + "train_accuracies.csv")
    network, optim, scheduler = read_net(root_dir + "model_parameters", image_size, output_size, device,
                                                class_weights)

    network = network.to(device)

    return network, optim, scheduler, len(train_losses), val_losses, train_losses, val_accuracies, train_accuracies