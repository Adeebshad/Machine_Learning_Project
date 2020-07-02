from arch_condVAE import CVAE
import torch,os,cv2,csv
import pandas as pd
import numpy as np
import torch.optim as optim
from skimage import io, transform
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import os

# temp = os.getcwd()
# temp1 = os.path.join(os.path.split(temp)[0],"layers")
# os.chdir(temp1)
# print(os.listdir())
# from loss import calculate_loss 
# from layers.dataloader import CustomDataset,ToTensor

# os.chdir(temp)

from layers.loss import calculate_loss
from layers.dataloader import CustomDataset,ToTensor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIM = 64*64*3   
LATENT_DIM = 256      
N_CLASSES = 73 
lr = 1e-3  
BATCH_SIZE=64
N_EPOCHS=10

imgpath = 'data/LFW/image'
# p = os.listdir(imgpath)
attrfile = 'data/LFW/processed_file.txt'



transformed_dataset = CustomDataset(csv_file=attrfile,
                                           root_dir=imgpath,
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))

train_iterator = DataLoader(transformed_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_iterator = DataLoader(transformed_dataset, batch_size=BATCH_SIZE)

model = CVAE(INPUT_DIM, LATENT_DIM, N_CLASSES)
optimizer = optim.Adam(model.parameters(), lr=lr)


def train():  
    model.train()
    train_loss = 0
    for i, sample in enumerate(train_iterator):
        x,y = sample['image'],sample['attributes']
        # print(x.shape,y.shape)
        x = x.view(-1,3,64,64)
        x = x.to(device)
        y = y.view(-1,73)
        y = y.to(device)
        
        optimizer.zero_grad()
        reconstructed_x, z_mu, z_var = model(x, y)
        loss = calculate_loss( reconstructed_x,x, z_mu, z_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print("TRain")
    return train_loss


def test():
    
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, sample  in enumerate(test_iterator):
            x,y = sample['image'].view(-1,3,64,64),sample['attributes']
            x = x.to(device)
            y = y.view(-1,73)
            y = y.to(device)
            reconstructed_x, z_mu, z_var = model(x, y)
            loss = calculate_loss(x, reconstructed_x, z_mu, z_var)
            test_loss += loss.item()
    print("test")

    return test_loss

  
    
for e in range(N_EPOCHS):

    train_loss = train()
    test_loss = test()

    train_loss /= len(train_iterator)
    test_loss /= len(test_iterator)

    print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')

    
