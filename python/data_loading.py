"""
File used for handling the data set
"""

from __future__ import print_function, division
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TrsF
from tqdm import tqdm


class data_set(Dataset):
    """
    class responsible for handling and dynamically retreiving data from the data set
    """
    def __init__(self, meta_path, root_dir, labels_path=False, transforms=None):
        """
        Init responsible for holding the list of filenames from which you can fetch data from
        :param meta_path: path to the metadata
        :param root_dir: path to the images files
        :param labels_path: path to the filenames and labels
        :param transforms: transforms to be applied to the data
        """

        self.train_image_dir = root_dir
        # self.metadata = pd.read_csv(meta_path) TODO: Use metadata
        self.file_names = os.listdir(self.train_image_dir)
        self.file_names.sort()
        self.transforms = transforms

        #  If its not the training data then don't add labels
        if labels_path:
            self.labels = pd.read_csv(labels_path)
            self.classes = self.labels.columns[1:10].values
        else:
            self.labels = False

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        """
        Dynamically loads and returns an image at specified index with label attached.
        If there is no label then it returns False as a label
        :param index: index of image to load
        :return: dictionary containing image and label
        """

        file_name = self.file_names[index]
        full_path = os.path.join(self.train_image_dir, file_name)
        image = Image.open(full_path)
        if self.labels is False:
            label = False
        else:
            label = self.get_class_name(self.labels.iloc[index].values)[0]

        if self.transforms:
            image = self.transforms(image)

        data = {'image': image, "label": label}

        return data

    def get_filename(self, index):
        return self.file_names[index]

    def get_label(self, index):
        """
        Returns the label as an integer at the specified index
        :param index:
        :return:
        """
        return self.get_class_name(self.labels.iloc[index].values)[0]

    def count_classes(self):

        """
        Counts the number of classes in the dataset and prints out the percentage of each class
        :return: a dictionary containing the class names and the number of times that class appears
        """

        LABELS = {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK', 4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC', 8: 'UNK'}
        labels_count = {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0, 'UNK': 0}
        print("Counting labels")
        for index in tqdm(range(len(self))):

            label = self.get_class_name(self.labels.iloc[index].values)[0]
            label = LABELS[label]
            try:
                labels_count[label] += 1
            except Exception as e:
                continue

        print(labels_count)

        for label, count in labels_count.items():
            print(f"{label}: {count/len(self) * 100}%")

        return labels_count

    def make_equal(self):
        """
        Turns the dataset into a binary classifier, only allows 4522 NV images and 4522 MEL images,
        then removes everything else
        :return:
        """

        print("Deleting Labels.")
        deleted = 0
        NV_count = 0

        for index in tqdm(range(len(self))):
            # need to remove the deleted files from the index, because of how pandas dataframes work
            index = index - deleted

            try:
                label_index = self.get_class_name(self.labels.iloc[index].values)[0]
            except Exception as e:
                print(self.labels)
                continue

            if label_index >= 2:
                self.file_names.remove(self.labels.iloc[index].values[0] + '.jpg')
                self.labels = self.labels.drop(index)
                deleted += 1
                self.labels = self.labels.reset_index(drop=True)

            elif label_index == 1 and NV_count >= 4522:
                self.file_names.remove(self.labels.iloc[index].values[0] + '.jpg')
                self.labels = self.labels.drop(index)
                deleted += 1
                self.labels = self.labels.reset_index(drop=True)

            elif label_index == 1:
                NV_count += 1

    def get_class_name(self, numbers):
        """
        returns the class names based on the number, i.e. 0 = MEL, 1 = NV... etc.
        :param numbers: a list of values, where one number is 1.0 to represent the class
        :return:
        """
        index = np.where(numbers == 1.0)
        return index[0] - 1

    def add_transforms(self, transforms):
        self.transforms = transforms


class RandomCrop(object):
    """
    Class used to randomly crop the image
    """

    def __init__(self, output_size):
        """
        Init takes in the desired crop size
        :param output_size:
        """

        # If output size is one parameter, make it a tuple of two of the same parameters
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        """
        Randomly crops off the side of the image to the specified size
        :param image: Image to be cropped
        :return: the cropped image
        """

        image = np.asarray(image)
        height, width = image.shape[:2]
        new_height, new_width = self.output_size

        top = np.random.randint(0, height - new_height)
        left = np.random.randint(0, width - new_width)

        image = image[top: top + new_height, left: left + new_width]

        image = Image.fromarray(image, 'RGB')

        return image


class RemoveBorders(object):

    def __init__(self, image_size, tolerance=0):
        self.tol = tolerance
        self.image_size = image_size
        self.index = 0

    def __call__(self, img):
        """
        Taken and adapted from https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
        :param image: Image to be cropped
        :return: the image with black borders removed
        """

        img.show()

        image = np.asarray(img)
        self.index = self.index + 1

        # Delete last 5 rows and columns
        image = np.delete(image, np.s_[self.image_size - 5:], 1)
        image = np.delete(image, np.s_[self.image_size - 5:], 0)

        image = np.delete(image, np.s_[:5], 1)
        image = np.delete(image, np.s_[:5], 0)

        prev_size = np.sum(image)

        mask = image > self.tol
        if image.ndim == 3:
            mask = mask.all(2)
        mask0, mask1 = mask.any(0), mask.any(1)
        image = image[np.ix_(mask0, mask1)]

        new_size = np.sum(image)

        image = Image.fromarray(image, 'RGB')

        image.show()

        if prev_size == new_size:
            return img
        else:
            return image

# Currently Unused
class RandomRotation(object):
    """
    Class used to randomly rotate the image
    """
    def __init__(self, angle):
        self.angles = angle

    def __call__(self, image):

        image = TrsF.rotate(image, np.random.choice(self.angles))

        return image
