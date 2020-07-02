import torch
import os
import cv2
import csv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
from skimage import io, transform
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.nn import Linear, ReLU, Tanh, CrossEntropyLoss, Sequential, Conv2d
from torch.nn import MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from torch.utils.data.sampler import SubsetRandomSampler
import dataloader
import test
import argparser
from model import DFCVAE


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    train_loss = 0

    for batch_idx, sample in enumerate(train_loader):
        data, y = sample['image'].to(device), sample['attributes'].to(device)
        data = data.view((-1, 3, 64, 64))
        y = y.view(-1, 3)
        optimizer.zero_grad()
        output, mu, logvar = model(data, y)
        loss = model.loss(output, data, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()),
                epoch, 100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader)
    print('Train set Average loss:', train_loss)
    return train_loss


def run_train(args):
    dataset = dataloader.CustomDataset(csv_file=args.csv_file,
                                       root_dir=args.root_dir,
                                       transform=transforms.Compose([
                                               dataloader.ToTensor()
                                           ]))

    validation_split = .006
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_iterator = torch.utils.data.DataLoader(dataset,
                                                 batch_size=args.batch_size,
                                                 sampler=train_sampler)
    test_iterator = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.batch_size,
                                                sampler=valid_sampler)

    model = DFCVAE(latent_size=args.latent_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs+1):
        train_loss = train(model, device, train_iterator, optimizer, epoch,
                           args.log_interval)
        test.test(model, device, test_iterator, epoch)


if __name__ == '__main__':
    args = argparser.train_parser()
    run_train(args)
