import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from skimage import io


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.attributes_frame = pd.read_csv(csv_file, delimiter=' ')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.attributes_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.attributes_frame.iloc[idx, 0])
        image = io.imread(img_name)
        image = cv2.resize(image, (64, 64))
        attributes = self.attributes_frame.loc[idx, ['Male', 'Eyeglasses',
                                                     'Smiling']]
        attributes = np.array([attributes])
        attributes = attributes.astype('float').reshape(-1, 1)
        sample = {'image': image, 'attributes': attributes}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, attributes = sample['image'], sample['attributes']
        norm_image = cv2.normalize(image, None, alpha=0, beta=1,
                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        im = torch.from_numpy(norm_image).float()
        ln = torch.from_numpy(attributes).float()
        return {'image': im,
                'attributes': ln}
