import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CarDataset(Dataset):
    """Car Dataset."""

    def __init__(self, images, labels, transform=None, discretize=False):
        """
        Arguments:
            images (array): Numpy array with the images.
            labels (array): Numpy array with the labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.discretize = discretize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.images[idx]
        y = self.labels[idx]

        x = torch.tensor(x, dtype=torch.float)  # Assuming images are in float format

        if self.transform:
            x, y = self.transform(x, y)
            if self.discretize:
                y = torch.tensor(y, dtype=torch.long)

        return x, y

