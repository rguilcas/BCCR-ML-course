import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset

# Define a canonical numbering of the classes
label_dict = {
    'dew': 0,
    'fogsmog': 1,
    'frost': 2,
    'glaze': 3,
    'hail': 4,
    'lightning': 5,
    'rain': 6,
    'rainbow': 7,
    'rime': 8,
    'sandstorm': 9,
    'snow': 10,
}

filepath = 'data/dataset/'
metapath = 'data/metadata/'

class WeatherDataset(Dataset):
    # Define a class to hold our data set
    # We will have a separate instance for each subset (train/val/test)
    def __init__(self, filepath, metapath, subset='train', D=32):
        # D is the pixel width and height that we change the images to
        self.filepath = filepath
        self.metapath = metapath
        self.D = D
        # We have predefined subsets for train/val/test
        if subset == 'test':
            group = 'test.csv'
        elif subset == 'val':
            group = 'val.csv'
        elif subset == 'train':
            group = 'train.csv'
        else:
            raise ValueError
        # The filenames for the images in the subset
        self.files = np.loadtxt(metapath+group, dtype=str)
        # To speed things up we will keep the resized images in memory
        self._loaded_images = {}
        self._labels = {}
        self._load_data()

    def __len__(self):
        # Length of our data set
        return len(self.files)

    def _load_data(self):
        for idx in range(len(self)):
            img, lab = self.__getitem__file(idx)
            self._loaded_images[idx] = img.numpy()
            self._labels[idx] = lab

    def __getitem__(self, idx):
        # During training we just use the correct image that's already loaded in memory
        return torch.tensor(self._loaded_images[idx]), self._labels[idx]

    def __getitem__file(self, idx):
        # Prepare an example image for the neural network
        # Returns the preprocessed image and its label
        # idx specifies which image in the order of the files list
        file_name = self.files[idx]
        # The label is the image class, e.g. snow
        label = file_name.split('/')[0]
        imgpath = self.filepath+file_name
        im = Image.open(imgpath)
        # Resize our image to make them smaller and consistently sized
        im = np.array(im.resize((self.D, self.D)))
        # Return the image in torch format, as well as the image label
        return torch.tensor(im, dtype=torch.float32).transpose(2, 0), label_dict[label]

def reset_seeds(seed=1):
    # Ensure reproducibility
    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)

def plot_learning_curves(train_curve, val_curve, title=None, baseline=None):
    # Make a plot showing how performance changes with each epoch on the
    # training and validation data
    plt.plot(train_curve, '-o', label='train')
    plt.plot(val_curve, '-o', label='val')
    if baseline is not None:
        plt.plot([0, len(train_curve)], [baseline]*2, 'k:', label='baseline')
    plt.legend()
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.title(title)

def get_class_name(class_num):
    return [ll for ll in label_dict if label_dict[ll] == class_num][0]