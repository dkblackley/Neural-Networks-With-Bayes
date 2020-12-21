from __future__ import print_function, division
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from tqdm import tqdm

class dataSet(Dataset):

    def __init__(self, meta_path, labels_path=False, transforms=None):

        self.train_image_dir = "ISIC_2019_Training_Input/"
        self.metadata = pd.read_csv(meta_path)
        self.file_names = os.listdir(self.train_image_dir)
        self.file_names.sort()
        self.transforms = transforms

        if labels_path:
            self.labels = pd.read_csv(labels_path)
            self.classes = self.labels.columns[1:10].values
        else:
            self.labels = False

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):

        file_name = self.file_names[index]

        full_path = os.path.join(self.train_image_dir, file_name)
        image = Image.open(full_path)
        label = self.get_class_name(self.labels.iloc[index].values)[0]

        if self.transforms:
            image = self.transforms(image)

        data = {'image': image, "label": label}

        return data

    def count_classes(self):

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

    # set NV to only have 4522 and remove everything that isn't MEL or NV
    def make_equal(self):

        print("Deleting Labels.")
        deleted = 0
        NV_count = 0

        for index in tqdm(range(len(self))):
            # if the file has already been deleted

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
        index = np.where(numbers == 1.0)
        return index[0] - 1

    def add_transforms(self, transforms):
        self.transforms = transforms



class randomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        image = np.asarray(image)

        height, width = image.shape[:2]
        new_height, new_width = self.output_size

        top = np.random.randint(0, height - new_height)
        left = np.random.randint(0, width - new_width)

        image = image[top: top + new_height, left: left + new_width]

        image = Image.fromarray(image, 'RGB')

        return image



class randomRotation(object):
    def __init__(self, angle):
        self.angles = angle

    def __call__(self, image):

        image = TF.rotate(image, np.random.choice(self.angles))

        return image
