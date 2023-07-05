import glob
import numpy as np
import os
from PIL import Image, ImageFile
import random
import torch
from torchvision import transforms


class RatioDataset(torch.utils.data.Dataset):

    def __init__(self, batch_size, percentage, *datasets):
        self.datasets = datasets
        self.batch_size = batch_size
        self.percentage = percentage

        self.ratio = int(batch_size * percentage)

    def __getitem__(self, idx):

        batch = idx // self.batch_size
        pos = idx % self.batch_size

        sketchy_idx = (self.ratio * batch) + pos
        ecommerce_idx = ((self.batch_size - self.ratio) * batch) + (pos - self.ratio)


        if (pos < self.ratio):
            #sketchy
            return self.datasets[0][sketchy_idx % len(self.datasets[0])]
        else:
            #ecommerce
            return self.datasets[1][ecommerce_idx % len(self.datasets[1])]

    def __len__(self):
        return sum([len(d) for d in self.datasets])
