import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class PreprocessingPipeline(nn.Module):
    def __init__(self, output_size=(224, 224), normalize=False, discretize=True, num_buckets=5):
        super(PreprocessingPipeline, self).__init__()

        self.pad = transforms.Pad((0, 84))
        self.resize = transforms.Resize(output_size)
        self.normalize = None
        self.discretize = discretize
        self.num_buckets = num_buckets

        # Normalization for the ImageNet dataset
        if normalize:
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _discretize_labels(self, labels):
        # Calculate buckets and bucket_labels dynamically
        buckets = np.linspace(0, 1, self.num_buckets + 1)
        bucket_labels = np.arange(self.num_buckets)

        bucketed_labels = np.digitize(labels, bins=buckets, right=False) - 1
        return np.array([bucket_labels[label] for label in bucketed_labels])

    def forward(self, image, labels=None):
        image = image.permute(2, 0, 1)
        image = self.pad(image)
        image = self.resize(image)
        if self.normalize:
            image = self.normalize(image)
        if self.discretize:
            labels = self._discretize_labels(labels)
        else: 
            labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels
    
import numpy as np
import torch.nn as nn
from torchvision import transforms

class PreprocessingPipelineCutOff(nn.Module):
    def __init__(self, output_size=(224, 224), normalize=False, discretize=True, cutoff=0.5):
        super(PreprocessingPipelineCutOff, self).__init__()

        self.pad = transforms.Pad((0, 84))
        self.resize = transforms.Resize(output_size)
        self.normalize = None
        self.discretize = discretize
        self.cutoff = cutoff

        if normalize:
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _discretize_labels(self, labels):
        bucketed_labels = np.where(labels < self.cutoff, 0, 1)
        return bucketed_labels

    def forward(self, image, labels=None):
        image = image.permute(2, 0, 1)
        image = self.pad(image)
        image = self.resize(image)
        labels = torch.tensor(labels, dtype=torch.float32)
        labels = labels.unsqueeze(-1)
        if self.normalize:
            image = self.normalize(image)
        if self.discretize:
            labels = self._discretize_labels(labels)
            labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels

class PreprocessingPipelineBinary(nn.Module):
    def __init__(self, output_size=(224, 224), normalize=False):
        super(PreprocessingPipelineBinary, self).__init__()

        self.pad = transforms.Pad((0, 84))
        self.resize = transforms.Resize(output_size)
        self.normalize = None

        if normalize:
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, image, labels=None):
        image = image.permute(2, 0, 1)
        image = self.pad(image)
        image = self.resize(image)
        labels = torch.tensor(labels, dtype=torch.float32)
        labels = labels.unsqueeze(-1)
        if self.normalize:
            image = self.normalize(image)
        return image, labels