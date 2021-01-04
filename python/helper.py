"""
Helper.py: File responsible for random helpful functions, such as; loading and saving data, calculating
means and standard deviation etc.
"""

import torch
import model
from tqdm import tqdm
import csv

LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC', 8: 'UNK'}


def plot_samples(data_set, data_plot):
    """
    displays image from data set
    :param data_set: data set to display images from
    :param data_plot: plotting class that uses matplotlib to display samples
    """

    for i in range(len(data_set)):
        data = data_set[i]
        data_plot.show_data(data)
        print(i, data['image'].size(), LABELS[data['label']])
        # Only display the first 3 images
        if i == 3:
            break

    for i_batch, sample_batch in enumerate(data_set):
        print(i_batch, sample_batch['image'].size(),
              LABELS[sample_batch['label']])

        if i_batch == 3:
            data_plot.show_batch(sample_batch, 3)
            break


def save_net(network, PATH):
    torch.save(network.state_dict(), PATH)


def load_net(PATH, image_size):
    net = model.Classifier(image_size)
    net.load_state_dict(torch.load(PATH))
    net.eval()
    return net

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
    print(f"Standard Deviation: {std}")

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