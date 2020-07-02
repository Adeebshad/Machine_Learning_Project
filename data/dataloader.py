import os
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from skimage import io


class CelebA(data.Dataset):
    """
    add docstring
    """
    def __init__(self, root_path, manifest_file_name,
                 image_container_name, image_size, selected_attrs):
        super(CelebA, self).__init__()
        self.root_path = root_path
        self.img_size = image_size
        self.manifest_file_name = manifest_file_name
        self.image_container_name = image_container_name

        attr_path = os.path.join(self.root_path, self.manifest_file_name)
        att_list = open(attr_path, 'r',
                        encoding='utf-8').readlines()[0].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        self.images = np.loadtxt(attr_path, skiprows=1, usecols=[0],
                                 dtype=np.str)
        self.labels = np.loadtxt(attr_path, skiprows=1, usecols=atts,
                                 dtype=np.int)

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.length = len(self.images)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_path, self.image_container_name,
                                self.images[index])
        image = io.imread(img_name)
        img = cv2.resize(image, (self.img_size, self.img_size))
        img = self.tf(img)
        att = torch.tensor((self.labels[index] + 1) // 2) # convert attribute value 1/0 from 1/-1
        return img, att

    def __len__(self):
        return self.length
