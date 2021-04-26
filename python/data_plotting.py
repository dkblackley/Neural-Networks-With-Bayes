"""
File used to plot various data with matplotlib
"""

from __future__ import print_function, division
import torch
import helper
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import seaborn as sn
import pandas as pd
from copy import deepcopy
from sklearn import metrics
import numpy as np

class DataPlotting:
    """
    Class for data plotting with matplotlib, contains methods for plotting loss, accuracy and
    confusion matrices
    """

    def __init__(self, data_loader, test_indexes, tick_size, font_size):
        self.data_loader = data_loader
        self.test_indexes = test_indexes

        self.colours = ['#1B2ACC', '#CC4F1B', '#228B22']
        self.edge_colours = ['#1B2ACC', '#CC4F1B', '#228B22']
        self.labels = ["Softmax response", "MC Dropout", "BBB"]

        self.LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC'}

        self.tick_size = tick_size
        self.font_size = font_size


    def show_data(self, data):
        """
        plots a single image
        :param data: dictionary containing image and label
        """
        image = data['image']
        if torch.is_tensor(image):
            trsfm = transforms.ToPILImage(mode='RGB')
            image = trsfm(image)

        plt.figure()
        plt.axis('off')
        plt.imshow(image)
        plt.title(f"{self.LABELS[data['label']]} Sample")
        plt.show()

    def show_batch(self, sample_batched, stop):
        """
        Shows a batch of images up to the specified stop value
        :param sample_batched: the batch of images and labels to plot
        :param stop: stop at this image in the batch
        """
        images_batch, labels_batch = \
         sample_batched['image'], sample_batched['label']

        for i in range(0, len(images_batch)):

            if i == stop:
                break

            image = images_batch[i]
            trsfm = transforms.ToPILImage(mode='RGB')
            image = trsfm(image)

            label = labels_batch[i].item()

            ax = plt.subplot(1, stop, i + 1)
            plt.imshow(image)
            plt.tight_layout()
            ax.set_title(f"{self.LABELS[label]} Sample")
            ax.axis('off')

        plt.ioff()
        plt.show()

    def plot_loss(self, save_dir, epochs, loss_values_val, loss_values_test):
        """
        Plots the learning curve
        :param epochs: number of epochs
        :param loss_values_val: list of loss values for the validation set
        :param loss_values_test: list of loss values for the testing set
        :return:
        """
        plt.plot(epochs, loss_values_test, label="Training set")
        plt.plot(epochs, loss_values_val, label="Validation set")
        plt.title("Loss over time")
        plt.xlabel("Epoch")
        plt.ylabel("Loss value")
        plt.legend(loc='best')
        plt.savefig(save_dir + "loss.png")
        plt.show()

    def plot_validation(self, save_dir, epochs, results_val, results_test):
        """
        Plots the accuracy over time
        :param epochs: epochs over which the data set was run
        :param results_val: list containing the results at each epoch for validation set
        :param results_test: list containing the results at each epoch for testing set
        """
        plt.plot(epochs, results_test, label="Training set")
        plt.plot(epochs, results_val, label="Validation set")
        plt.title("Accuracy over time")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='best')

        maxi = max([max(results_test) + 10, max(results_val) + 10])
        mini = min([min(results_test) - 10, min(results_val) - 10])

        plt.ylim([mini, maxi])
        plt.savefig(save_dir + "accuracy.png")
        plt.show()

    def plot_confusion(self, array, root_dir, title):
        """
        Plots a confusion matrix
        :param title: title for the plot
        :param array: a list of lists containing a representation of how the network predicted
        """

        values = list(self.LABELS.values())

        if isinstance(array[0][0], int):
            form = 'd'
        else:
            form = 'g'

        df_cm = pd.DataFrame(array, index=[i for i in list(values)],
                             columns=[i for i in list(values)])
        plt.figure(figsize=(15, 10))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt=form, cmap="YlGnBu") # font size
        plt.title(title)

        plt.xlabel("Predicted label")
        plt.ylabel("True label")

        plt.savefig(f"{root_dir + title}.png")
        plt.show()

    def count_sampels_in_intervals(self, predictions, root_dir, title, bins, skip_first=False, color='#1B2ACC'):
        """
        Counts how many predictions occur within given probability range for the calibration plot
        :param predictions: the list of predictions
        :param root_dir: the directory to save the plot to
        :param title: the title of the plot
        :param bins: the range of probabilities to look in
        :param skip_first: can be difficult to read the plot due to the class imbalance, set this to true to skip
        the first bin, making the plot readable
        :return:
        """

        bin_count = []
        class_probabilities = np.delete(deepcopy(predictions), -1, axis=1)

        for i in range(0, 8):
            current_probabilities = class_probabilities[:, i: i + 1]
            current_count = []

            for c in range(1, bins + 1):
                if skip_first and c == 1:
                    continue

                total = len(current_probabilities[
                    (current_probabilities <= c / bins) & (current_probabilities > c / bins - 1 / bins)])

                current_count.append(total)
            bin_count.append(current_count)

        figure, axs = plt.subplots(2, 4, figsize=(20, 10))
        titles = list(self.LABELS.values())
        figure.suptitle(title)
        # add a big axis, hide frame
        figure.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Probability", fontsize=self.font_size)
        plt.ylabel("Number of Samples", fontsize=self.font_size)
        axs = axs.ravel()
        probabilities_range = []

        for i in range(1, bins + 1):
            if skip_first and i == 1:
                continue

            probabilities_range.append(f"{round(i / bins - 1/bins, 1)} to {round(i / bins, 1)}")

        for idx, a in enumerate(axs):
            a.tick_params(axis="x", labelsize=self.tick_size)
            a.tick_params(axis="y", labelsize=self.tick_size)
            a.bar(probabilities_range, bin_count[idx], alpha=0.5, color=color)
            a.set_title(titles[idx])

        plt.tight_layout()
        plt.savefig(f"{root_dir + title}.png")
        plt.show()

    def plot_calibration(self, predictions, title, root_dir, bins):
        """
        Plots a reliability diagram that shows how well calibrated the probabilities are by class
        :param predictions: the list of probability distributions
        :param title: The title of the plot
        :param root_dir: The directory to save the plot in
        :param bins: the range of probabilities you want to test, i.e a bins of 3 will give 3 points on the diagram
        :return:
        """

        average_probs = []
        relative_freq = []

        for k in range(0, len(predictions)):

            average_probs.append([])
            relative_freq.append([])

            prediction = deepcopy(predictions[k])
            class_probabilities = np.delete(prediction, -1, axis=1)  # delete the uncertainty estimation

            for i in tqdm(range(0, 8)):
                current_average = []
                current_freq = []
                current_probabilities = class_probabilities[:, i: i+1]  # splice out only the class that we're concerned with

                # Calculate how many probabilities appear within range
                for c in range(1, bins + 1):
                    average = current_probabilities[
                                               (current_probabilities <= c/bins) & (current_probabilities > c/bins - 1/bins)].mean()

                    # Catch cases where there is no value in the bin
                    if np.isnan(average):
                        continue

                    current_average.append(average)
                    correct = 0
                    total = 0

                    # Calculate how many actual correct answers occur within this range
                    for j in range(0, len(current_probabilities)):
                        if current_probabilities[j] <= c/bins and current_probabilities[j] > c/bins - 1/bins:
                            if helper.is_prediction_corect(i, self.test_indexes[j], self.data_loader):
                                correct += 1
                            total += 1

                    if correct == 0:
                        current_freq.append(0)
                    else:
                        current_freq.append(correct/total)

                average_probs[k].append(current_average)
                relative_freq[k].append(current_freq)

        figure, axs = plt.subplots(2, 4, figsize=(20, 10))
        titles = list(self.LABELS.values())
        figure.suptitle(title)
        # add a big axis, hide frame
        figure.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Average Probability", fontsize=self.font_size)
        plt.ylabel("Relative frequency of positive samples", fontsize=self.font_size)
        axs = axs.ravel()

        for idx, a in enumerate(axs):
            a.tick_params(axis="x", labelsize=14)
            a.tick_params(axis="y", labelsize=14)
            for i in range(0, len(predictions)):
                a.plot(average_probs[i][idx], relative_freq[i][idx], marker="s", color=self.colours[i], label=self.labels[i])
            if idx == 0:
                a.legend(loc='best')
            a.plot([0.0, 1.0], [0.0, 1.0], linestyle='dashed', color="black")
            a.set_title(titles[idx])

        plt.tight_layout()
        plt.savefig(f"{root_dir + title}.png")
        plt.show()

    def plot_risk_coverage(self, predictions, root_dir, title, cost_matrx=False):
        """
        Plots a risk coverage curve, showing risk in % on the y-axis showing the risk that the predicitions might be
        wrong and coverage % on the x-axis that plots the % of the dataset that has been included to get that risk
        :param array: list of accuracy, should be a list containing all predicitions on each class and also the
        entropy value as the last item in the array.
        :param title: Title of the plot
        :param cost_matrx: whether or not to apply a cost matrix
        :return:
        """
        accuracy = []
        uncertainties = []
        answers = []
        coverage = []
        predictions = deepcopy(predictions)

        for i in range(0, len(predictions)):
            accuracy.append([])
            uncertainties.append([])
            # add the uncertainty metrics to a seperate list
            for pred in predictions[i]:
                uncertainties[i].append(pred.pop())
            predictions[i] = np.array(predictions[i])
            uncertainties[i] = np.array(uncertainties[i])
            # get all the answers in one list for easy removal later
            answers.append(self.data_loader.get_all_labels(self.test_indexes))

        # Remove the highest uncertainty and then remove that prediction from the answers and continually re-get average accuracy
        for i in tqdm(range(0, len(self.test_indexes))):
            coverage.append(i/len(self.test_indexes))
            for c in range(0, len(predictions)):
                total = 0
                correct = 0
                for k in range(0, len(predictions[c])):
                    predic = predictions[c][k].argmax()
                    if predictions[c][k].argmax() == answers[c][k]:
                        total += 1
                        correct += 1
                    else:
                        total += 1

                accuracy[c].append(correct/total)

                max_uncertainty_row = np.unravel_index(uncertainties[c].argmax(), uncertainties[c].shape)[0]
                predictions[c] = np.delete(predictions[c], max_uncertainty_row, axis=0)
                uncertainties[c] = np.delete(uncertainties[c], max_uncertainty_row, axis=0)
                del answers[c][max_uncertainty_row]

        coverage.reverse()

        softmax_AUC = round(metrics.auc(coverage, accuracy[0]), 3)
        dropout_AUC = round(metrics.auc(coverage, accuracy[1]), 3)
        BBB_AUC = round(metrics.auc(coverage, accuracy[2]), 3)

        print("Softmax AUC: " + str(softmax_AUC))
        print("MC_dropout AUC: " + str(dropout_AUC))
        print("BBB AUC: " + str(BBB_AUC))

        plt.plot(coverage, accuracy[0], label="Softmax response")
        plt.plot(coverage, accuracy[1], label="MC Dropout")
        plt.plot(coverage, accuracy[2], label="BbB")
        plt.title(title)
        #plt.ylim(0.7, 1.1)
        plt.xlabel("Coverage")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        """txt_string = '\n'.join((
            f'Softmax Response AUC: {softmax_AUC}',
            f'MC Dropout AUC: {dropout_AUC}'
        ))"""
        props = dict(boxstyle='round', fc='white', alpha=0.5)
        #plt.text(0.6, 5, txt_string, bbox=props, fontsize=self.font_size)

        plt.savefig(f"{root_dir + title}.png")
        plt.show()

    def plot_cost_coverage(self, costs, root_dir, title, uncertainty=False):
        """
        Plots a single LEC by coverage curve, the average LEC is calculated then the highest LEC prediction
        is removed and average LEC is calculated again
        :param costs: a list of the Expected cost predictions, expects the softmx then MC then BbB
        :param root_dir: directory to save everything to
        :param title: Title of the plot
        :param uncertainty: Whether or not a uncertainty metric is to be considered with predictions
        :return:
        """

        average_cost = []
        coverage = []
        costs_sorted_by_entropy = []
        costs_sorted = []
        # set up all the arrays
        for i in range(0, len(costs)):
            costs[i] = np.array(costs[i])
            average_cost.append([])
            costs_sorted_by_entropy.append([])
            costs_sorted.append([])
            if uncertainty:
                # Sort the arrays by entropy to make removal easier
                ind = np.argsort(costs[i][:, -1])
                costs_sorted_by_entropy[i] = costs[i][ind]
                costs_sorted_by_entropy[i] = np.delete(costs_sorted_by_entropy[i], -1, axis=1)

                # put just the predictions in costs_sorted
                for c in range(0, len(self.test_indexes)):

                    costs_sorted[i].append(float(costs_sorted_by_entropy[i][c].min()))
            else:
                # sort the costs by highest LEC
                for c in range(0, len(self.test_indexes)):
                    lowest = np.unravel_index(costs[i].argmin(), costs[i].shape)
                    costs_sorted[i].append(float(costs[i][lowest]))
                    costs[i] = np.delete(costs[i], lowest[0], axis=0)

        for i in tqdm(range(0, len(self.test_indexes))):
            for c in range(0, len(costs)):
                average_cost[c].append(sum(costs_sorted[c])/len(costs_sorted[c]))

                # Remove highest cost and go again
                costs_sorted[c].pop()

            coverage.append(1 - i/len(self.test_indexes))

        for c in range(0, len(costs)):
            average_cost[c].reverse()

        coverage.reverse()


        #dropout_AUC = round(metrics.auc(coverage, average_cost[0]), 3)
        #softmax_AUC = round(metrics.auc(coverage, average_cost[1]), 3)


        plt.plot(coverage, average_cost[0], label="Softmax response")
        plt.plot(coverage, average_cost[1], label="MC Dropout")
        plt.plot(coverage, average_cost[2], label="BbB")
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.title(title)
        plt.xlabel("Coverage")
        plt.ylabel("Average LEC")
        plt.legend(loc='best')
        """txt_string = '\n'.join((
            f'SR AUC: {softmax_AUC}',
            f'MC Dropout AUC: {dropout_AUC}'
        ))"""
        #props = dict(boxstyle='round', fc='white', alpha=0.5)
        #plt.text(5, 0.6, txt_string, bbox=props, fontsize=self.font_size)

        plt.savefig(f"{root_dir + title}.png")
        plt.show()

    def plot_true_cost_coverage(self, predictions, root_dir, title, uncertainty=False, costs=True, n_classes=8, flatten=False):
        """
        Plots test cost by coverage, using either the LEC as a prediction or raw probabilities
        :param preds: the list of predictions
        :param root_dir: the directory to save the graph
        :param title: the title of the plot
        :param uncertainty: whether or not to include uncertainty
        :param costs: whether to use LEC of raw probabilities
        :param n_classes:
        :param flatten: whether to use the usual cost matrix or a cost matrix of all 1's
        :return:
        """

        # set up initial arrays
        values = []
        averages = []
        new_preds = []
        preds = []


        for i in range(0, len(predictions)):
            preds.append([])
            if not uncertainty and not costs:
                new_pred = deepcopy(predictions[i])
                for p in new_pred:
                    p.pop()
                preds[i] = np.array(new_pred)
            else:
                preds[i] = np.array(predictions[i])
            values.append([])
            averages.append([])
            new_preds.append([])


        coverage = []
        for i in range(0, len(self.test_indexes)):

            true_label = self.data_loader.get_label(self.test_indexes[i])
            # Find the costs of our predictions
            if costs:
                for c in range(0, len(preds)):
                    values[c].append(helper.find_true_cost(np.argmin(preds[c][i]), true_label, flatten=flatten, uncertain=uncertainty))
                    new_preds[c].append(np.min(preds[c][i]))

            else:
                for c in range(0, len(preds)):
                    values[c].append(helper.find_true_cost(np.argmax(preds[c][i]), true_label, flatten=flatten, uncertain=uncertainty))
                    new_preds[c].append(np.max(preds[c][i]))

        # set the LEC to be the only prediction and average the test cost
        for i in range(0, len(preds)):
            preds[i] = np.array(new_preds[i])
            averages[i].append(sum(values[i]) / len(values[i]))

        coverage.append(0.0)

        # Remove the highest LEC and then remove that cost from the test costs and continually re-get average test cost
        for i in range(0, len(self.test_indexes)):
            if costs:
                for c in range(0, len(preds)):
                    max_row = np.unravel_index(preds[c].argmax(), preds[c].shape)[0]
                    preds[c] = np.delete(preds[c], max_row, axis=0)
                    del values[c][max_row]

            else:
                for c in range(0, len(preds)):
                    min_row = np.unravel_index(preds[c].argmin(), preds[c].shape)[0]
                    preds[c] = np.delete(preds[c], min_row, axis=0)
                    del values[c][min_row]

            coverage.append((i + 1)/len(self.test_indexes))
            for c in range(0, len(preds)):

                if len(values[c]) == 0:
                    averages[c].append(0)
                else:
                    averages[c].append(sum(values[c]) / len(values[c]))

        softmax_AUC = round(metrics.auc(coverage, averages[0]), 3)
        dropout_AUC = round(metrics.auc(coverage, averages[1]), 3)
        BBB_AUC = round(metrics.auc(coverage, averages[2]), 3)

        print("Softmax AUC: " + str(softmax_AUC))
        print("MC_dropout AUC: " + str(dropout_AUC))
        print("BBB AUC: " + str(BBB_AUC))

        coverage.reverse()
        plt.plot(coverage, averages[0], label="Softmax Response")
        plt.plot(coverage, averages[1], label="MC Dropout")
        plt.plot(coverage, averages[2], label="BbB")
        plt.title(title)
        plt.xlabel("Coverage")
        plt.ylabel("Average Test Cost")
        plt.legend(loc='best')

        plt.savefig(f"{root_dir + title}.png")
        plt.show()

    def plot_true_cost_coverage_by_class(self, predictions, root_dir, title, costs=True, n_classes=8, flatten=False):
        """
        Plots the test cost coverage by class, cycles over each classification and finds the average test cost of all
        classifications, then removes the highest LEC or lowest probability and re-gets the Average test cost. continues
        to do this until the entire set has been covered.
        :param preds: List of predictions, expects a added uncertainty metric prediction if costs is False
        :param root_dir: Directory to save plots to
        :param title: Title of the plot
        :param costs: Whether or not we are using probabilities applied to a cost matrix or raw probabilities
        :param n_classes: the number of classes
        :param flatten: Set to True to use a cost matrix with cost 1 applied everywhere except the diagonal
        :return:
        """

        # Hold the entropy/uncertainty metrics
        preds_copy = []
        preds = []

        for i in range(0, len(predictions)):
            preds.append([])

        # Remove the uncertainty estimation from the raw probabilities and changes the array to numpy arrays
        if not costs:
            for i in range(0, len(predictions)):
                preds_copy.append(deepcopy(predictions[i]))

                for c in range(0, len(predictions[i])):
                    preds_copy[i][c].pop()

                preds[i] = np.array(preds_copy[i])

        else:

            for i in range(0, len(predictions)):
                preds[i] = np.array(predictions[i])

        results = []
        results_average = []
        new_preds = []

        # Attach dictionaries for by class classification
        for i in range(0, len(preds)):
            results.append({'MEL': [], 'NV': [], 'BCC': [], 'AK': [], 'BKL': [], 'DF': [], 'VASC': [], 'SCC': []})
            results_average.append({'MEL': [], 'NV': [], 'BCC': [], 'AK': [], 'BKL': [], 'DF': [], 'VASC': [], 'SCC': []})
            new_preds.append({'MEL': [], 'NV': [], 'BCC': [], 'AK': [], 'BKL': [], 'DF': [], 'VASC': [], 'SCC': []})

        # Gets the indexes for where each specific class occurs in the test data
        for i in range(0, len(preds)):
            preds[i], label_indexes = helper.get_label_indexes(preds[i], self.test_indexes, self.data_loader)

        coverage = {}
        # Creates coverage for x-axis matplotlib plot.
        for key in results[0]:
            coverage[key] = [i/len(label_indexes[key]) for i in range(0, len(label_indexes[key]) + 1)]
            coverage[key].reverse()

        # Cycles over all the predictions and creates a list of just the prediction, to make it easier to remove later
        for key in label_indexes.keys():
            indexes = label_indexes[key]

            for i in range(0, len(indexes)):
                true_label = self.data_loader.get_label(indexes[i])

                for c in range(0, len(preds)):

                    # Add only the current class probability to new_preds
                    if costs:
                        results[c][key].append(
                            helper.find_true_cost(np.argmin(preds[c][key][i]), true_label, flatten=flatten))
                        new_preds[c][key].append(np.min(preds[c][key][i]))

                    else:
                        results[c][key].append(
                            helper.find_true_cost(np.argmax(preds[c][key][i]), true_label, flatten=flatten))
                        new_preds[c][key].append(np.max(preds[c][key][i]))

        # Set preds to be new_preds, just for ease of wording, also append the first result at coverage = 1
        for i in range(0, len(preds)):

            for key in new_preds[i]:
                preds[i][key] = np.array(new_preds[i][key])

            for key in results_average[i].keys():
                results_average[i][key].append(sum(results[i][key])/len(results[i][key]))

        """Cycles over each key and gets the index of the highest LEC or the lowest probability, removes the prediction
        at that index and the result at that index, gets average test cost again, then repeats until all predictions 
        have been removed."""
        for key in label_indexes.keys():
            indexes = label_indexes[key]

            for i in range(0, len(indexes)):
                for c in range(0, len(preds)):
                    if costs:
                        max_row = np.unravel_index(preds[c][key].argmax(), preds[c][key].shape)[0]
                        preds[c][key] = np.delete(preds[c][key], max_row, axis=0)
                        del results[c][key][max_row]

                    else:
                        min_row = np.unravel_index(preds[c][key].argmin(), preds[c][key].shape)[0]
                        preds[c][key] = np.delete(preds[c][key], min_row, axis=0)
                        del results[c][key][min_row]

                    if len(preds[c][key]) == 0:
                        results_average[c][key].append(0)

                    else:
                        results_average[c][key].append(sum(results[c][key]) / len(results[c][key]))


        figure, axs = plt.subplots(2, 4, figsize=(20, 10))
        titles = list(results[0].keys())
        figure.suptitle(title)
        # add a big axis, hide frame
        figure.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Coverage")
        plt.ylabel("Average Test Cost")
        axs = axs.ravel()

        for idx, a in enumerate(axs):

            a.tick_params(axis="x", labelsize=self.tick_size)
            a.tick_params(axis="y", labelsize=self.tick_size)

            a.plot(coverage[self.LABELS[idx]], results_average[0][self.LABELS[idx]], label="Softmax Response")
            a.plot(coverage[self.LABELS[idx]], results_average[1][self.LABELS[idx]], label="MC Dropout")
            a.plot(coverage[self.LABELS[idx]], results_average[2][self.LABELS[idx]], label="BbB")
            a.set_title(titles[idx])

            if idx == 3:
                a.legend(loc='best')
            # a.set_xlabel(xaxes)
            # a.set_ylabel(yaxes)

        plt.tight_layout()
        plt.savefig(f"{root_dir + title}.png")
        plt.show()

    def plot_correct_incorrect_uncertainties(self, correct, incorrect, root_dir, title, by_class=False, prediction_index=1):
        """
        Plots a histogram showing the number of correct predictions and the number of Incorrect predictions alongside
        their uncertainty metric
        :param correct: list of correct predictions
        :param incorrect: list of incorrect predictions
        :param root_dir: the directory to save the figure
        :param title: Title of the plot
        :param by_class: whether the histogram should encapsulate all classes, or should be broken into subplots by class
        :param prediction_index: Where the prediction in the correct or incorrect list, used to seperate uncertainty
        metrics from the predictions
        :return:
        """
        correct_preds = []
        incorrect_preds = []

        if by_class:

            labels_correct = {'MEL': [], 'NV': [], 'BCC': [], 'AK': [], 'BKL': [], 'DF': [], 'VASC': [], 'SCC': []}
            labels_incorrect = {'MEL': [], 'NV': [], 'BCC': [], 'AK': [], 'BKL': [], 'DF': [], 'VASC': [], 'SCC': []}

            # find and add the uncertainty metric into the dicts
            for pred in correct:
                labels_correct[self.LABELS[pred[prediction_index]]].append(pred[-1])

            for pred in incorrect:
                labels_incorrect[self.LABELS[pred[prediction_index]]].append(pred[-1])

            figure, axs = plt.subplots(2, 4, figsize=(20, 10))
            titles = list(labels_incorrect.keys())
            figure.suptitle(title)
            # add a big axis, hide frame
            figure.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axis
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel("Uncertainty", fontsize=self.font_size)
            plt.ylabel("Number of Samples",  fontsize=self.font_size)
            axs = axs.ravel()

            for idx, a in enumerate(axs):

                a.tick_params(axis="x", labelsize=self.tick_size)
                a.tick_params(axis="y", labelsize=self.tick_size)

                a.hist(labels_correct[self.LABELS[idx]], alpha=0.5, label="Correct")
                a.hist(labels_incorrect[self.LABELS[idx]], alpha=0.5, label="Incorrect")
                a.set_title(titles[idx])
                if idx == 3:
                    a.legend(loc='best')
                # a.set_xlabel(xaxes)
                # a.set_ylabel(yaxes)

            plt.tight_layout()
            plt.savefig(f"{root_dir + title}.png")
            plt.show()

        else:
            for pred in correct:
                correct_preds.append(pred[-1])

            for pred in incorrect:
                incorrect_preds.append(pred[-1])

            plt.hist(correct_preds, alpha=0.5, label="Correct")
            plt.hist(incorrect_preds, alpha=0.5, label="Incorrect")

            plt.xlabel("Uncertainty")
            plt.ylabel("Samples")
            plt.legend(loc='best')

            plt.title(title)

            plt.savefig(f"{root_dir + title}.png")
            plt.show()

    def average_uncertainty_by_class(self, correct, incorrect, root_dir, title):
        """
        Plots a scatter plot showing accuracy on the y axis and entropy on the x axis
        :param correct: The correct predictions
        :param incorrect: the Incorrect predictions
        :param root_dir: directory to save plot to
        :param title: The title of the plot
        :return:
        """

        labels_accuracy = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}
        labels_entropy = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}
        labels_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}

        correct = deepcopy(correct)
        incorrect = deepcopy(incorrect)

        # Count all the answers and add their entropy to the dicts
        for i in range(0, len(correct)):
            answer = correct[i][0]
            entropy = correct[i][-1]

            labels_accuracy[self.LABELS[answer]] += 1
            labels_entropy[self.LABELS[answer]] += entropy
            labels_count[self.LABELS[answer]] += 1

        for i in range(0, len(incorrect)):
            answer = incorrect[i][0]
            entropy = incorrect[i][-1]

            labels_entropy[self.LABELS[answer]] += entropy
            labels_count[self.LABELS[answer]] += 1

        accuracy = []
        entropy = []

        # Calculate accuracy
        for key in labels_accuracy.keys():
            accuracy.append((labels_accuracy[key]/labels_count[key]) * 100)
            entropy.append((labels_entropy[key]/labels_count[key]))


        labels = list(labels_accuracy.keys())

        fig, ax = plt.subplots()
        ax.scatter(entropy, accuracy,
                   color=['Black', 'Blue', 'Brown', 'Crimson', 'DarkGreen', 'DarkMagenta', 'Gray', 'Peru'], s=20)

        for i, txt in enumerate(labels):
            ax.annotate(txt, (entropy[i], accuracy[i]), xytext=(entropy[i], accuracy[i] + 1))

        plt.xlabel("Average Entropy")
        plt.ylabel("Accuracy")
        plt.ylim([min(accuracy) - 5, max(accuracy) + 5])
        plt.xlim(left=0)
        plt.title(title)
        plt.savefig(f"{root_dir + title}.png")
        plt.show()

    def plot_each_mc_pass(self, mc_dir, BBB_dir, predictions_softmax, test_indexes, test_data, save_dir, title, cost_matrix=False):

        accuracy = [[], [], []]

        pos_avg_entropy = [[], [], []]
        neg_avg_entropy = [[], [], []]


        for i in tqdm(range(0, 100)):
            current_mc_predictions = helper.read_rows(mc_dir + f"entropy/mc_forward_pass_{i}_" + "entropy" + ".csv")
            current_mc_predictions = helper.string_to_float(current_mc_predictions)

            current_BBB_predictions = helper.read_rows(BBB_dir + f"entropy/BBB_forward_pass_{i}_" + "entropy" + ".csv")
            current_BBB_predictions = helper.string_to_float(current_BBB_predictions)

            entropy = [[], [], []]

            for c in range(0, len(current_mc_predictions)):
                entropy[0].append(predictions_softmax[c][-1])
                entropy[1].append(current_mc_predictions[c][-1])
                entropy[2].append(current_BBB_predictions[c][-1])

            correct = [[], [], []]
            incorrect = [[], [], []]
            uncertain = [[], [], []]

            correct[0], incorrect[0], uncertain[0] = helper.get_correct_incorrect(predictions_softmax, test_data,
                                                                                  test_indexes, cost_matrix)

            correct[1], incorrect[1], uncertain[1] = helper.get_correct_incorrect(current_mc_predictions, test_data,
                                                                                      test_indexes, cost_matrix)

            correct[2], incorrect[2], uncertain[2] = helper.get_correct_incorrect(current_BBB_predictions, test_data,
                                                                                  test_indexes, cost_matrix)

            for c in range(0, len(accuracy)):
                accuracy[c].append((len(correct[c]) / len(test_indexes)) * 100)

                pos_avg_entropy[c].append(
                    (sum(entropy[c]) / len(entropy[c])) + (len(correct[c]) / len(test_indexes)) * 100)

                neg_avg_entropy[c].append(
                    (sum(entropy[c]) / len(entropy[c]) * -1) + (len(correct[c]) / len(test_indexes)) * 100)


        passes = [i for i in range(0, 100)]
        for i in range(0, len(accuracy)):
            if i == 0:
                plt.plot(passes, accuracy[i], '--', color=self.colours[i], label=self.labels[i])
            else:
                plt.plot(passes, accuracy[i], color=self.colours[i], label=self.labels[i])
            plt.fill_between(passes, pos_avg_entropy[i], neg_avg_entropy[i], alpha=0.5, edgecolor=self.edge_colours[i])

        plt.legend(loc='lower right')
        plt.xlabel("Forward Passes")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.savefig(f"{save_dir + title}.png")
        plt.show()

    def plot_each_mc_true_cost(self, costs_softmax, save_dir, mc_dir, BBB_dir, title):

        avg_costs = [[], [], []]
        passes = []

        costs_softmax = deepcopy(costs_softmax)

        total = 0
        for i in range(0, len(self.test_indexes)):
            true_label = self.data_loader.get_label(self.test_indexes[i])
            total += helper.find_true_cost(np.argmin(costs_softmax[i]), true_label)
        sr_avg_cost = total/len(costs_softmax)

        for i in tqdm(range(0, 100)):
            current_costs = [[], []]

            current_costs[0] = helper.read_rows(mc_dir + f"costs/mc_forward_pass_{i}_costs.csv")
            helper.remove_last_row(current_costs[0])
            current_costs[0] = helper.string_to_float(current_costs[0])

            current_costs[1] = helper.read_rows(BBB_dir + f"costs/BBB_forward_pass_{i}_costs.csv")
            helper.remove_last_row(current_costs[1])
            current_costs[1] = helper.string_to_float(current_costs[1])

            passes.append(i)
            avg_costs[0].append(sr_avg_cost)

            for c in range(0, len(current_costs)):

                total = 0

                for i in range(0, len(self.test_indexes)):
                    true_label = self.data_loader.get_label(self.test_indexes[i])
                    total += helper.find_true_cost(np.argmin(current_costs[c][i]), true_label)

                avg = total/len(current_costs[c])
                avg_costs[c + 1].append(avg)

        for i in range(0, len(avg_costs)):
            if i == 0:
                plt.plot(passes, avg_costs[i], '--', color=self.colours[i], label=self.labels[i])
            else:
                plt.plot(passes, avg_costs[i], color=self.colours[i], label=self.labels[i])
            #plt.fill_between(passes, pos_avg_entropy_mc, neg_avg_entropy_mc, alpha=0.5, edgecolor='#1B2ACC')


        plt.legend(loc='upper right')
        plt.xlabel("Forward Passes")
        plt.ylabel("Average Test cost")
        plt.title(title)
        plt.savefig(f"{save_dir + title}.png")
        plt.show()