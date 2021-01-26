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

# LABELS = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AK': 3, 'BKL': 4, 'DF': 5, 'VASC': 6, 'SCC': 7}


class DataPlotting:
    """
    Class for data plotting with matplotlib, contains methods for plotting loss, accuracies and
    confusion matrices
    """

    def __init__(self, unknown):
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

        plt.xlabel("True label")
        plt.ylabel("Predicted label")

        plt.savefig(f"{root_dir + title}.png")
        plt.show()

    def plot_risk_coverage(self, predictions_mc_original, predictions_softmax_original, root_dir, title, test_data, test_indexs):
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


        for i in range(0, len(predictions_softmax)):
            entropies_sm.append(predictions_softmax[i].pop())
            entropies_mc.append(predictions_mc[i].pop())

        predictions_mc = deepcopy(predictions_mc_original)
        predictions_softmax = deepcopy(predictions_softmax_original)

        entropies_mc.sort()
        entropies_sm.sort()

        sm_accuracies.append(0)
        mc_accuracies.append(0)

        coverage.append(0.0)

        for i in tqdm(range(0, len(test_indexs))):
            correct_mc, incorrect_mc, uncertain_mc = helper.get_correct_incorrect(predictions_mc, test_data,
                                                                                  test_indexs, threshold=entropies_mc.pop())

            correct_sr, incorrect_sr, uncertain_mc = helper.get_correct_incorrect(predictions_softmax, test_data,
                                                                                  test_indexs, threshold=entropies_sm.pop())

            if incorrect_sr and correct_sr:
                sm_accuracies.append((len(incorrect_sr) / len(test_indexs)) * 100)
            else:
                sm_accuracies.append(0)

            if correct_mc and incorrect_mc:
                mc_accuracies.append((len(incorrect_mc) / len(test_indexs)) * 100)
            else:
                mc_accuracies.append(0)

            coverage.append((i/len(test_indexs)))



        helper.write_csv(mc_accuracies, root_dir + "MC_risk_coverage.csv")
        helper.write_csv(sm_accuracies, root_dir + "SM_risk_coverage.csv")


        plt.plot(coverage, mc_accuracies, label="MC Dropout")
        plt.plot(coverage, sm_accuracies, label="Softmax response")
        maxi = max([max(mc_accuracies) + 10, max(mc_accuracies) + 10])
        plt.ylim([0, maxi])
        plt.title(title)
        plt.xlabel("Coverage")
        plt.ylabel("Innaccuracy (%)")
        plt.legend(loc='best')

        plt.savefig(f"{root_dir + title}.png")
        plt.show()

    def plot_correct_incorrect_uncertainties(self, correct, incorrect, root_dir, title):

        correct_preds = []
        incorrect_preds = []
        correct = deepcopy(correct)
        incorrect = deepcopy(incorrect)
        i = 0

        for pred in correct:
            correct_preds.append(pred.pop())
            i = i + 1

        for pred in incorrect:
            incorrect_preds.append(pred.pop())
            i = i + 1

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
            entropy = correct[i][1]

            labels_accuracy[self.LABELS[answer]] += 1
            labels_entropies[self.LABELS[answer]] += entropy
            labels_count[self.LABELS[answer]] += 1

        for i in range(0, len(incorrect)):
            answer = incorrect[i][0]
            entropy = incorrect[i][1]

            labels_entropies[self.LABELS[answer]] += entropy
            labels_count[self.LABELS[answer]] += 1

        accuracies = []
        entropies = []

        for key in labels_accuracy.keys():
            accuracies.append((labels_accuracy[key]/labels_count[key]) * 100)
            entropies.append((labels_entropies[key]/labels_count[key]))


        labels = list(labels_accuracy.keys())

        fig, ax = plt.subplots()
        ax.scatter(entropies, accuracies)

        for i, txt in enumerate(labels):
            ax.annotate(txt, (entropies[i], accuracies[i]))

        plt.xlabel("Average Entropy")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.savefig(f"{root_dir + title}.png")
        plt.show()