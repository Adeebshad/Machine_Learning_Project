import os
import cv2
import json
import torch
import argparse
from skimage import io
from argparser import *
from os.path import join
from model.model import AttGAN
import torch.utils.data as data
from data.dataloader import CelebA
import torchvision.utils as vutils
from utils.utils import Progressbar,find_model
from utils.utils import check_attribute_conflict
import torchvision.transforms as transforms

# Bald Bangs Black_Hair Blond_Hair Brown_Hair Bushy_Eyebrows Eyeglasses Male Mouth_Slightly_Open Mustache No_Beard Pale_Skin Young

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transforms = transforms.Compose([        
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
def preprocess(image,attr,size):
    """
    This function preprocess custom input images .
    
    Params:
    
    -- image       : root path of your system as string.
    -- attr        : Directory name where images store in root_path
    -- size        : test attributes file name

    
    Returns:
    preprocessed image and its 13 attributes
    
    """
    
    img = cv2.resize(image, (size,size))
#     print(img.shape,attr.shape)
    img = data_transforms(img)
    img_a = img.view((1,3,384,384))
    att_a = attr.view((1,13))
    
    return img_a,att_a

    
def inference(root_path,image_folder_name,image_name,args):
    """
    [Bald Bangs Black_Hair Blond_Hair Brown_Hair Bushy_Eyebrows Eyeglasses Male Mouth_Slightly_Open Mustache No_Beard Pale_Skin Young]
    
    This function train the AttGAN model and validate.It generates sample images from valid dataset.
    For this You need to pass the 13 existing attributes of input image and then pass the list of index from 13 attributes
    that you want to change.After running the inference file it will generate images by changing that given.
    
    Params:
    
    --root_path               : root path of your system as string.
    -- image_container_name   : Directory name where images store in root_path
    -- image_file_name        : Image file name
    -- weights_path           : directory name of weights 
    -- weights_name           : name of the weights as string
    -- input_img_attr         : 13 attributes of input image as list
    -- index                  : list of index that you want to change
    -- thres_int              : float, it will use to normalize attributes value in specific threshold value
    -- img_size               : size of image 
    
    
    Returns:
    None
    
    """
    img_attr = args.input_img_attr
    value = args.index
    print(img_attr,value)
    
    if type(args.input_img_attr[0]) is str:
        img_attr=[int(i) for i in args.input_img_attr]
        value = [int(i) for i in args.index]
    
    
    for i in value:
        if img_attr[i-1]==0:
            img_attr[i-1]=1
        else:
            img_attr[i-1]=0
    print(img_attr)       
    output_path = join('output')
    os.makedirs(output_path, exist_ok=True)
    image = io.imread(join(root_path,image_folder_name,image_name))
    attr = torch.tensor(img_attr)
    index = torch.tensor(value) 
    
    attr = (attr * 2 - 1) * args.thres_int
    attr = attr.type(torch.float)
#     attr[index-1]=torch.tensor([1.])
    
    img_a,attr = preprocess(image,attr,args.img_size)
    img_a = img_a.to(device) 
    attr = attr.to(device) 
    print(img_a.shape,attr.shape)
    
    attgan = AttGAN()
    attgan.load(join(args.root_path,args.weights_path))
    progressbar = Progressbar()
    attgan.eval()

    with torch.no_grad():   
        
        p = attgan.G(img_a, attr)
        vutils.save_image(
            p, join(output_path, image_name),
            nrow=1, normalize=True, range=(-1., 1.)
        )
        print("Image save in output directory!")

        
if __name__=="__main__":
    args = inference_parser()
    print(args)
    inference(args.root_path,args.image_folder_name,args.image_fname,args)

    
    