from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import numpy as np
import seaborn as sn
import pandas as pd


#LABELS = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AK': 3, 'BKL': 4, 'DF': 5, 'VASC': 6, 'SCC': 7, 'UNK': 8}
LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC', 8: 'UNK'}

class dataPlotting():

    def show_data(self, data):

        image = data['image']
        if torch.is_tensor(image):
            trsfm = transforms.ToPILImage(mode='RGB')
            image = trsfm(image)
            #image = image.numpy()
            #image = image.transpose((2, 1, 0))

        plt.figure()
        plt.axis('off')
        plt.imshow(image)
        plt.title(f"{LABELS[data['label']]} Sample")
        plt.show()

    # Helper function to show images from a batch
    def show_batch(self, sample_batched, stop):
        images_batch, labels_batch = \
         sample_batched['image'], sample_batched['label']

        im_size = images_batch.size(2)

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
            ax.set_title(f"{LABELS[label]} Sample")
            ax.axis('off')

        plt.ioff()
        plt.show()

    def plot_loss(self, loss_cycle, loss_values_val, loss_values_test):
        plt.plot(loss_cycle, loss_values_test, label="Training set")
        plt.plot(loss_cycle, loss_values_val, label="Validation set")
        plt.title("Loss over time")
        plt.xlabel("Epoch")
        plt.ylabel("Loss value")
        plt.legend(loc='best')
        plt.show()

    def plot_validation(self, epochs, results_val, results_test):
        plt.plot(epochs, results_test, label="Training set")
        plt.plot(epochs, results_val, label="Validation set")
        plt.title("Accuracy over time")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc='best')

        maxi = max([max(results_test) + 20, max(results_val) + 20])
        mini = min([min(results_test) - 20, min(results_val) - 20])

        plt.ylim([mini, maxi])
        plt.show()

    def plot_confusion(self, array):



        print(array)
        df_cm = pd.DataFrame(array, index=[i for i in list(LABELS.values())],
                             columns=[i for i in list(LABELS.values())])
        plt.figure(figsize=(15, 10))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}, fmt='d') # font size

        plt.show()