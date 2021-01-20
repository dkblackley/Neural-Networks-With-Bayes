"""
File used to plot various data with matplotlib
"""

from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import seaborn as sn
import pandas as pd
from copy import deepcopy

# LABELS = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AK': 3, 'BKL': 4, 'DF': 5, 'VASC': 6, 'SCC': 7}


class DataPlotting:
    """
    Class for data plotting with matplotlib, contains methods for plotting loss, accuracies and
    confusion matrices
    """
    LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC', 8: 'UNK'}

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

    def plot_confusion(self, array, title):
        """
        Plots a confusion matrix
        :param title: title for the plot
        :param array: a list of lists containing a representation of how the network predicted
        """

        values = list(self.LABELS.values())
        values.pop()

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

        plt.savefig(f"saved_model/{title}.png")
        plt.show()

    def plot_risk_coverage(self, predictions_mc, predictions_softmax, title):
        """
        Plots a risk coverage curve, showing risk in % on the y-axis showing the risk that the predicitions might be
        wrong and coverage % on the x-axis that plots the % of the dataset that has been included to get that risk
        :param array: list of accuracies, should be a list containing all predicitions on each class and also the
        entropy value as the last item in the array.
        :param title: Title of the plot
        :return:
        """
        predictions_mc = deepcopy(predictions_mc)
        predictions_softmax = deepcopy(predictions_softmax)
        mc_entropies = []
        sm_entropies = []
        coverage = []
        i = 0
        for array in predictions_mc:
            i = i + 1
            mc_entropies.append(array.pop())
            coverage.append(i/len(predictions_mc))

        for array in predictions_softmax:
            sm_entropies.append(array.pop())

        mc_entropies.sort()
        sm_entropies.sort()

        plt.plot(coverage, mc_entropies, label="MC Dropout")
        plt.plot(coverage, sm_entropies, label="Softmax response")
        plt.title(title)
        plt.xlabel("Coverage")
        plt.ylabel("Risk")
        plt.legend(loc='best')

        plt.savefig("saved_model/risk_curve.png")
        plt.show()

    def plot_correct_incorrect_uncertainties(self, correct, incorrect, title):

        predictions = []
        correct = deepcopy(correct)
        incorrect = deepcopy(incorrect)
        i = 0

        for pred in correct:
            predictions.append([pred.pop(), i/(len(correct) + len(incorrect)), "Correct"])
            i = i + 1

        for pred in incorrect:
            predictions.append([pred.pop(), i/(len(correct) + len(incorrect)), "Incorrect"])
            i = i + 1

        correct_incorrect = pd.DataFrame(predictions, columns=['Entropy', "Density", "Classification"])

        sn.displot(correct_incorrect, x="Entropy", hue="Classification", kind="kde", fill=True)
        plt.title(title)

        plt.savefig(f"saved_model/{title}.png")
        plt.show()

    def average_uncertainty_by_class(self, correct, incorrect, title):

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
        plt.savefig(f"saved_model/{title}.png")
        plt.show()