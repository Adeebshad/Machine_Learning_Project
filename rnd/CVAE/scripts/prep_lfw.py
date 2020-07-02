import torch,os,cv2,csv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 202599
scale = 64
attrDim = 40

imgpath = 'data/img_align_celeba/'
attrfile = 'data/LFW/'

file = open(attrfile+'LFW_processed_file.txt', 'r') 
print("processing...")
for line in file:
    lines = line.split()
    str1 = " "  
    s = (str1.join(lines)) 
    with open(attrfile+'processed_file.txt', 'a') as the_file:
        the_file.write(s)
        the_file.write('\n')
print("DONE")
landmarks_frame = pd.read_csv(attrfile+'processed_file.txt',delimiter=' ')
print("Total number of attributes :",landmarks_frame.shape)
print("Total number of images :",len(os.listdir(imgpath)))




