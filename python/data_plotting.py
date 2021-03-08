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

# LABELS = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AK': 3, 'BKL': 4, 'DF': 5, 'VASC': 6, 'SCC': 7}


class DataPlotting:
    """
    Class for data plotting with matplotlib, contains methods for plotting loss, accuracies and
    confusion matrices
    """

    def __init__(self, unknown, data_loader, test_indexes):
        self.data_loader = data_loader
        self.test_indexes = test_indexes
        if unknown:
            self.LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'UNK'}
        else:
            self.LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC'}

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

    def plot_loss(self, epochs, loss_values_val, loss_values_test):
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
        plt.savefig("saved_model/loss.png")
        plt.show()

    def plot_validation(self, epochs, results_val, results_test):
        """
        Plots the accuracies over time
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
        plt.savefig("saved_model/accuracy.png")
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

    def count_sampels_in_intervals(self, predictions, root_dir, title, bins, skip_first=False):

        bin_count = []

        class_probabilites = np.delete(predictions, -1, axis=1)



        for i in range(0, 8):
            current_probabilites = class_probabilites[:, i: i + 1]
            current_count = []

            for c in range(1, bins + 1):

                if skip_first and c == 1:
                    continue

                total = len(current_probabilites[
                    (current_probabilites <= c / bins) & (current_probabilites > c / bins - 1 / bins)])

                current_count.append(total)

            bin_count.append(current_count)


        figure, axs = plt.subplots(2, 4, figsize=(20, 10))
        titles = list(self.LABELS.values())
        figure.suptitle(title)
        # add a big axis, hide frame
        figure.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Probability", fontsize=16)
        plt.ylabel("Number of Samples", fontsize=16)
        axs = axs.ravel()

        probabilities_range = []


        for i in range(1, bins + 1):

            if skip_first and i == 1:
                continue
            probabilities_range.append(f"{round(i / bins - 1/bins, 1)} to {round(i / bins, 1)}")

        for idx, a in enumerate(axs):
            a.bar(probabilities_range, bin_count[idx], alpha=0.5)
            a.set_title(titles[idx])

            # a.set_xlabel(xaxes)
            # a.set_ylabel(yaxes)

        plt.tight_layout()
        plt.savefig(f"{root_dir + title}.png")
        plt.show()

    def plot_calibration(self, predictions, title, root_dir, bins):

        average_probs = []
        relative_freq = []

        class_probabilites = np.delete(predictions, -1, axis=1)

        for i in tqdm(range(0, 8)):
            current_average = []
            current_freq = []
            current_probabilites = class_probabilites[:,i: i+1]

            for c in range(1, bins + 1):

                average = current_probabilites[
                                           (current_probabilites <= c/bins) & (current_probabilites > c/bins - 1/bins)].mean()

                if np.isnan(average):
                    continue

                current_average.append(average)
                correct = 0
                total = 0
                for j in range(0, len(current_probabilites)):
                    if current_probabilites[j] <= c/bins and current_probabilites[j] > c/bins - 1/bins:
                        if helper.is_prediction_corect(i, self.test_indexes[j], self.data_loader):
                            correct += 1
                        total += 1

                if correct == 0:
                    current_freq.append(0)
                else:
                    current_freq.append(correct/total)


            average_probs.append(current_average)
            relative_freq.append(current_freq)




        figure, axs = plt.subplots(2, 4, figsize=(20, 10))
        titles = list(self.LABELS.values())
        figure.suptitle(title)
        # add a big axis, hide frame
        figure.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Average Probability", fontsize=16)
        plt.ylabel("Relative frequency of positive samples", fontsize=16)
        axs = axs.ravel()

        for idx, a in enumerate(axs):
            a.plot(average_probs[idx], relative_freq[idx], marker="s")
            a.plot([0.0, 1.0], [0.0, 1.0], linestyle='dashed', color="black")
            a.set_title(titles[idx])
            #if idx == 3:
            #    a.legend(loc='best')
            # a.set_xlabel(xaxes)
            # a.set_ylabel(yaxes)

        plt.tight_layout()
        plt.savefig(f"{root_dir + title}.png")
        plt.show()

    def plot_risk_coverage(self, predictions_mc_original, predictions_softmax_original, root_dir, title, load=False, cost_matrx=False):
        """
        Plots a risk coverage curve, showing risk in % on the y-axis showing the risk that the predicitions might be
        wrong and coverage % on the x-axis that plots the % of the dataset that has been included to get that risk
        :param array: list of accuracies, should be a list containing all predicitions on each class and also the
        entropy value as the last item in the array.
        :param title: Title of the plot
        :return:
        """
        predictions_mc = deepcopy(predictions_mc_original)
        predictions_softmax = deepcopy(predictions_softmax_original)
        mc_accuracies = []
        sm_accuracies = []
        coverage = []
        entropies_mc = []
        entropies_sm = []

        if load:
            mc_accuracies = helper.read_csv(root_dir + title + "_mc.csv")
            sm_accuracies = helper.read_csv(root_dir + title + "_sr.csv")

            coverage = [i/len(self.test_indexes) for i in range(0, len(self.test_indexes))]
            coverage.append(1.0)

        else:
            for i in range(0, len(predictions_softmax)):
                entropies_sm.append(predictions_softmax[i][-1])
                entropies_mc.append(predictions_mc[i][-1])

            entropies_mc.sort()
            entropies_sm.sort()

            sm_accuracies.append(0)
            mc_accuracies.append(0)

            coverage.append(0.0)

            for i in tqdm(range(0, len(self.test_indexes))):
                correct_mc, incorrect_mc, uncertain_mc = helper.get_correct_incorrect(predictions_mc, self.data_loader,
                                                                                      self.test_indexes, cost_matrx, threshold=entropies_mc.pop())

                correct_sr, incorrect_sr, uncertain_mc = helper.get_correct_incorrect(predictions_softmax, self.data_loader,
                                                                                      self.test_indexes, cost_matrx, threshold=entropies_sm.pop())

                if incorrect_sr and correct_sr:
                    sm_accuracies.append((len(incorrect_sr) / len(self.test_indexes)) * 100)
                else:
                    sm_accuracies.append(0)

                if correct_mc and incorrect_mc:
                    mc_accuracies.append((len(incorrect_mc) / len(self.test_indexes)) * 100)
                else:
                    mc_accuracies.append(0)

                coverage.append((i/len(self.test_indexes)))



        helper.write_csv(mc_accuracies, root_dir + title + "_mc.csv")
        helper.write_csv(sm_accuracies, root_dir + title + "_sr.csv")

        dropout_AUC = round(metrics.auc(coverage, mc_accuracies), 3)
        softmax_AUC = round(metrics.auc(coverage, sm_accuracies), 3)


        plt.plot(coverage, mc_accuracies, label="MC Dropout")
        plt.plot(coverage, sm_accuracies, label="Softmax response")
        maxi = max([max(mc_accuracies) + 10, max(mc_accuracies) + 10])
        plt.ylim([0, maxi])
        plt.title(title)
        plt.xlabel("Coverage")
        plt.ylabel("100 - Accuracy")
        plt.legend(loc='best')
        txt_string = '\n'.join((
            f'Softmax Response AUC: {softmax_AUC}',
            f'MC Dropout AUC: {dropout_AUC}'
        ))
        props = dict(boxstyle='round', fc='white', alpha=0.5)
        plt.text(0.6, 5, txt_string, bbox=props, fontsize=10)

        plt.savefig(f"{root_dir + title}.png")
        plt.show()

    def plot_cost_coverage(self, costs_mc_original, costs_softmax_original, root_dir, title, uncertainty=False, n_classes=8, load=False, cost_matrx=False):
        """
        Plots a risk coverage curve, showing risk in % on the y-axis showing the risk that the predicitions might be
        wrong and coverage % on the x-axis that plots the % of the dataset that has been included to get that risk
        :param array: list of accuracies, should be a list containing all predicitions on each class and also the
        entropy value as the last item in the array.
        :param title: Title of the plot
        :return:
        """

        costs_mc = np.array(costs_mc_original)
        costs_sr = np.array(costs_softmax_original)
        mc_average_costs = []
        sr_average_costs = []
        coverage = []

        if uncertainty:
            ind = np.argsort(costs_mc[:, -1])
            costs_mc_sorted_by_entropy = costs_mc[ind]

            ind = np.argsort(costs_sr[:, -1])
            costs_sr_sorted_by_entropy = costs_sr[ind]

            costs_sr_sorted_by_entropy = np.delete(costs_sr_sorted_by_entropy, -1, axis=1)
            costs_mc_sorted_by_entropy = np.delete(costs_mc_sorted_by_entropy, -1, axis=1)

            costs_mc_sorted = []
            costs_sr_sorted = []

            for i in range(0, len(self.test_indexes)):
                costs_sr_sorted.append(float(costs_sr_sorted_by_entropy[i].min()))
                costs_mc_sorted.append(float(costs_mc_sorted_by_entropy[i].min()))


        else:
            costs_mc_sorted = []
            costs_sr_sorted = []

            for i in range(0, len(self.test_indexes)):
                lowest = np.unravel_index(costs_sr.argmin(), costs_sr.shape)
                costs_sr_sorted.append(float(costs_sr[lowest]))
                costs_sr = np.delete(costs_sr, lowest[0], axis=0)

                lowest = np.unravel_index(costs_mc.argmin(), costs_mc.shape)
                costs_mc_sorted.append(float(costs_mc[lowest]))
                costs_mc = np.delete(costs_mc, lowest[0], axis=0)



        for i in tqdm(range(0, len(self.test_indexes))):

            mc_average_costs.append(sum(costs_mc_sorted)/len(costs_mc_sorted))
            sr_average_costs.append(sum(costs_sr_sorted)/len(costs_sr_sorted))

            # Remove highest cost and go again
            costs_mc_sorted.pop()
            costs_sr_sorted.pop()

            coverage.append(1 - i/len(self.test_indexes))



        mc_average_costs.reverse()
        sr_average_costs.reverse()

        coverage.reverse()

        dropout_AUC = round(metrics.auc(coverage, mc_average_costs), 3)
        softmax_AUC = round(metrics.auc(coverage, sr_average_costs), 3)


        plt.plot(coverage, mc_average_costs, label="MC Dropout")
        plt.plot(coverage, sr_average_costs, label="Softmax response")
        maxi = max([max(mc_average_costs) + 3, max(mc_average_costs) + 3])
        plt.ylim([0, maxi])
        plt.title(title)
        plt.xlabel("Coverage")
        plt.ylabel("Average LEC")
        plt.legend(loc='best')
        txt_string = '\n'.join((
            f'SR AUC: {softmax_AUC}',
            f'MC Dropout AUC: {dropout_AUC}'
        ))
        props = dict(boxstyle='round', fc='white', alpha=0.5)
        plt.text(5, 0.6, txt_string, bbox=props, fontsize=10)

        plt.savefig(f"{root_dir + title}.png")
        plt.show()

    def plot_true_cost_coverage(self, preds, root_dir, title, uncertainty=False, costs=True, n_classes=8, flatten=False):

        values = []
        averages = []
        new_preds = []

        for pred in preds:
            pred = np.array(pred)
            values.append([])
            averages.append([])
            new_preds.append([])

        #index_copy = deepcopy(self.test_indexes)

        coverage = []

        for i in range(0, len(self.test_indexes)):

            true_label = self.data_loader.get_label(self.test_indexes[i])

            if costs:

                for c in range(0, len(preds)):

                    values[c].append(helper.find_true_cost(np.argmin(preds[c][i]), true_label, flatten=flatten, uncertain=uncertainty))
                    new_preds[c].append(np.min(preds[c][i]))

            else:

                for c in range(0, len(preds)):

                    values[c].append(helper.find_true_cost(np.argmax(preds[c][i]), true_label, flatten=flatten, uncertain=uncertainty))
                    new_preds[c].append(np.max(preds[c][i]))

        for i in range(0, len(preds)):
            preds[i] = np.array(new_preds[i])
            averages[i].append(sum(values[i]) / len(values[i]))

        coverage.append(0.0)

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


        coverage.reverse()


        #dropout_AUC = round(metrics.auc(coverage, mc_average), 3)
        #softmax_AUC = round(metrics.auc(coverage, sr_average), 3)


        plt.plot(coverage, averages[0], label="MC Dropout")
        plt.plot(coverage, averages[1], label="Softmax response")
        #maxi = max([max(mc_average) + 2, max(mc_average) + 2])
        #plt.ylim([0, maxi])
        plt.title(title)
        plt.xlabel("Coverage")
        plt.ylabel("Average Test Cost")
        plt.legend(loc='best')

        plt.savefig(f"{root_dir + title}.png")
        plt.show()

    def plot_true_cost_coverage_by_class(self, preds, root_dir, title, uncertainty=False, costs=True, n_classes=8, flatten=False):

        entropies = []
        preds_copy = []

        if uncertainty:
            labels = {'MEL': [], 'NV': [], 'BCC': [], 'AK': [], 'BKL': [], 'DF': [], 'VASC': [], 'SCC': []}
        else:
            labels = {'MEL': [], 'NV': [], 'BCC': [], 'AK': [], 'BKL': [], 'DF': [], 'VASC': [], 'SCC': [], 'UNK': []}

        """for i in range(0, len(preds)):
            entropies.append({'MEL': [], 'NV': [], 'BCC': [], 'AK': [], 'BKL': [], 'DF': [], 'VASC': [], 'SCC': []})"""


        if not costs:

            for i in range(0, len(preds)):

                preds_copy.append(deepcopy(preds[i]))

                for c in range(0, len(preds[i])):
                    entropies.append(preds[i][c].pop())

                preds[i] = np.array(preds[i])


        else:

            for i in range(0, len(preds)):
                preds[i] = np.array(preds[i])
        #index_copy = deepcopy(self.test_indexes)

        results = []
        results_average = []
        new_preds = []


        for i in range(0, len(preds)):
            results.append({'MEL': [], 'NV': [], 'BCC': [], 'AK': [], 'BKL': [], 'DF': [], 'VASC': [], 'SCC': []})
            results_average.append({'MEL': [], 'NV': [], 'BCC': [], 'AK': [], 'BKL': [], 'DF': [], 'VASC': [], 'SCC': []})
            new_preds.append({'MEL': [], 'NV': [], 'BCC': [], 'AK': [], 'BKL': [], 'DF': [], 'VASC': [], 'SCC': []})

        for i in range(0, len(preds)):
            preds[i], label_indexes = helper.get_label_indexes(preds[i], self.test_indexes, self.data_loader)

        coverage = {}

        for key in results[0]:
            coverage[key] = [i/len(label_indexes[key]) for i in range(0, len(label_indexes[key]) + 1)]
            coverage[key].reverse()


        for key in label_indexes.keys():

            indexes = label_indexes[key]

            for i in range(0, len(indexes)):

                true_label = self.data_loader.get_label(indexes[i])

                for c in range(0, len(preds)):

                    if costs:
                        results[c][key].append(
                            helper.find_true_cost(np.argmin(preds[c][key][i]), true_label, flatten=flatten, uncertain=uncertainty))
                        new_preds[c][key].append(np.min(preds[c][key][i]))

                    else:

                        results[c][key].append(
                            helper.find_true_cost(np.argmax(preds[c][key][i]), true_label, flatten=flatten, uncertain=uncertainty))
                        new_preds[c][key].append(np.max(preds[c][key][i]))

        for i in range(0, len(preds)):

            for key in new_preds[i]:
                preds[i][key] = np.array(new_preds[i][key])

            for key in results_average[i].keys():
                results_average[i][key].append(sum(results[i][key])/len(results[i][key]))


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




        #dropout_AUC = round(metrics.auc(coverage, mc_average), 3)
        #softmax_AUC = round(metrics.auc(coverage, sr_average), 3)

        figure, axs = plt.subplots(2, 4, figsize=(20, 10))
        titles = list(results[0].keys())
        figure.suptitle(title)
        # add a big axis, hide frame
        figure.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Coverage", fontsize=16)
        plt.ylabel("Average Test Cost", fontsize=16)
        axs = axs.ravel()

        for idx, a in enumerate(axs):

            a.plot(coverage[self.LABELS[idx]], results_average[0][self.LABELS[idx]], label="MC Dropout")
            a.plot(coverage[self.LABELS[idx]], results_average[1][self.LABELS[idx]], label="Softmax Response")
            a.set_title(titles[idx])

            if idx == 3:
                a.legend(loc='best')
            # a.set_xlabel(xaxes)
            # a.set_ylabel(yaxes)

        plt.tight_layout()
        plt.savefig(f"{root_dir + title}.png")
        plt.show()

    def plot_correct_incorrect_uncertainties(self, correct, incorrect, root_dir, title, by_class=False, prediction_index=1):

        correct_preds = []
        incorrect_preds = []

        if by_class:

            labels_correct = {'MEL': [], 'NV': [], 'BCC': [], 'AK': [], 'BKL': [], 'DF': [], 'VASC': [], 'SCC': []}
            labels_incorrect = {'MEL': [], 'NV': [], 'BCC': [], 'AK': [], 'BKL': [], 'DF': [], 'VASC': [], 'SCC': []}

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
            plt.xlabel("Entropy", fontsize=16)
            plt.ylabel("Number of Samples", fontsize=16)
            axs = axs.ravel()

            for idx, a in enumerate(axs):

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

            plt.xlabel("Entropy")
            plt.ylabel("Samples")
            plt.legend(loc='best')

            plt.title(title)

            plt.savefig(f"{root_dir + title}.png")
            plt.show()

    def average_uncertainty_by_class(self, correct, incorrect, root_dir, title):

        labels_accuracy = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}
        labels_entropies = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}
        labels_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}

        correct = deepcopy(correct)
        incorrect = deepcopy(incorrect)

        for i in range(0, len(correct)):
            answer = correct[i][0]
            entropy = correct[i][-1]

            labels_accuracy[self.LABELS[answer]] += 1
            labels_entropies[self.LABELS[answer]] += entropy
            labels_count[self.LABELS[answer]] += 1

        for i in range(0, len(incorrect)):
            answer = incorrect[i][0]
            entropy = incorrect[i][-1]

            labels_entropies[self.LABELS[answer]] += entropy
            labels_count[self.LABELS[answer]] += 1

        accuracies = []
        entropies = []

        for key in labels_accuracy.keys():
            accuracies.append((labels_accuracy[key]/labels_count[key]) * 100)
            entropies.append((labels_entropies[key]/labels_count[key]))


        labels = list(labels_accuracy.keys())

        fig, ax = plt.subplots()
        ax.scatter(entropies, accuracies,
                   color=['Black', 'Blue', 'Brown', 'Crimson', 'DarkGreen', 'DarkMagenta', 'Gray', 'Peru'], s=20)

        for i, txt in enumerate(labels):
            ax.annotate(txt, (entropies[i], accuracies[i]), xytext=(entropies[i], accuracies[i] + 1))

        plt.xlabel("Average Entropy")
        plt.ylabel("Accuracy")
        plt.ylim([min(accuracies) - 5, max(accuracies) + 5])
        plt.title(title)
        plt.savefig(f"{root_dir + title}.png")
        plt.show()

    def plot_each_mc_pass(self, mc_dir, predictions_softmax, test_indexes, test_data, save_dir, title, uncertainty, cost_matrix=False):

        mc_accuracies = []
        sm_accuracies = []

        pos_avg_entropies_mc = []
        neg_avg_entropies_mc = []
        pos_avg_entropies_sm = []
        neg_avg_entropies_sm = []

        for i in tqdm(range(0, 100)):
            current_mc_predictions = helper.read_rows(mc_dir + uncertainty + '/' + f"mc_forward_pass_{i}_" + "entropy" + ".csv")
            current_mc_predictions = helper.string_to_float(current_mc_predictions)

            entropies_sm = []
            entropies_mc = []

            for c in range(0, len(current_mc_predictions)):
                entropies_mc.append(current_mc_predictions[c][-1])
                entropies_sm.append(predictions_softmax[c][-1])

            correct_mc, incorrect_mc, uncertain_mc = helper.get_correct_incorrect(current_mc_predictions, test_data,
                                                                                      test_indexes, cost_matrix)

            correct_sr, incorrect_sr, uncertain_mc = helper.get_correct_incorrect(predictions_softmax, test_data,
                                                                                      test_indexes,cost_matrix)

            sm_accuracies.append((len(correct_sr) / len(test_indexes)) * 100)
            mc_accuracies.append((len(correct_mc) / len(test_indexes)) * 100)

            pos_avg_entropies_mc.append((sum(entropies_mc)/len(entropies_mc)) + (len(correct_mc) / len(test_indexes)) * 100)
            pos_avg_entropies_sm.append((sum(entropies_sm)/len(entropies_sm)) + (len(correct_sr) / len(test_indexes)) * 100)

            neg_avg_entropies_sm.append((sum(entropies_sm) / len(entropies_sm) * -1) + (len(correct_sr) / len(test_indexes)) * 100)
            neg_avg_entropies_mc.append((sum(entropies_mc)/len(entropies_mc) * -1) + (len(correct_mc) / len(test_indexes)) * 100)


        passes = [i for i in range(0, 100)]

        plt.plot(passes, mc_accuracies, color='#1B2ACC', label="MC Dropout")
        plt.fill_between(passes, pos_avg_entropies_mc, neg_avg_entropies_mc, alpha=0.5, edgecolor='#1B2ACC')

        plt.plot(passes, sm_accuracies, '--', color='#CC4F1B', label="Softmax response")
        plt.fill_between(passes, pos_avg_entropies_sm, neg_avg_entropies_sm, alpha=0.5, edgecolor='#CC4F1B')

        plt.legend(loc='lower right')
        plt.xlabel("Forward Passes")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.savefig(f"{save_dir + title}.png")
        plt.show()

    def plot_each_mc_cost(self, predictions_softmax, save_dir, title, uncertainty):

        mc_costs = []
        sr_costs = []

        entropies_sm = []

        pos_avg_entropies_mc = []
        neg_avg_entropies_mc = []
        pos_avg_entropies_sm = []
        neg_avg_entropies_sm = []

        cost_softmax = helper.read_rows(save_dir + "softmax_costs.csv")
        cost_softmax = helper.attach_last_row(cost_softmax, predictions_softmax)
        cost_softmax = helper.string_to_float(cost_softmax)

        for i in range(0, len(cost_softmax)):
            entropies_sm.append(cost_softmax[i].pop())

        total = 0
        for cost in cost_softmax:
            total += min(cost)

        average_lowest_cost_sr = total/len(cost_softmax)

        for i in tqdm(range(0, 100)):
            current_mc_predictions = helper.read_rows(save_dir + uncertainty + '/' + f"mc_forward_pass_{i}_" + uncertainty + ".csv")
            current_mc_costs = helper.read_rows(save_dir + 'costs/' + f"mc_forward_pass_{i}_costs.csv")
            current_mc_costs = helper.attach_last_row(current_mc_costs, current_mc_predictions)
            current_mc_costs = helper.string_to_float(current_mc_costs)



            entropies_mc = []

            for c in range(0, len(current_mc_predictions)):
                entropies_mc.append(current_mc_costs[c].pop())


            total = 0
            for cost in current_mc_costs:
                total += min(cost)

            average_lowest_cost_mc = total/len(current_mc_costs)


            mc_costs.append(average_lowest_cost_mc)
            sr_costs.append(average_lowest_cost_sr)

            """pos_avg_entropies_mc.append((sum(entropies_mc)/len(entropies_mc)) + mc_costs[-1])
            pos_avg_entropies_sm.append((sum(entropies_sm)/len(entropies_sm)) + sr_costs[-1])

            neg_avg_entropies_mc.append((sum(entropies_mc)/len(entropies_mc) * -1) + mc_costs[-1])
            neg_avg_entropies_sm.append((sum(entropies_sm) / len(entropies_sm) * -1) + sr_costs[-1])"""


        passes = [i for i in range(0, 100)]

        plt.plot(passes, mc_costs, color='#1B2ACC', label="MC Dropout")
        #plt.fill_between(passes, pos_avg_entropies_mc, neg_avg_entropies_mc, alpha=0.5, edgecolor='#1B2ACC')

        plt.plot(passes, sr_costs, '--', color='#CC4F1B', label="Softmax response")
        #plt.fill_between(passes, pos_avg_entropies_sm, neg_avg_entropies_sm, alpha=0.5, edgecolor='#CC4F1B')

        plt.legend(loc='lower right')
        plt.xlabel("Forward Passes")
        plt.ylabel("Average lowest expected cost")
        plt.title(title)
        plt.savefig(f"{save_dir + title}.png")
        plt.show()