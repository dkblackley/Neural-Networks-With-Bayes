from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, utils


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

    def plot_loss(self, loss_values, loss_cycle):
        plt.plot(loss_cycle, loss_values)
        plt.title("Loss over time")
        plt.xlabel("Epoch")
        plt.ylabel("Loss value")
        plt.show()