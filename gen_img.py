import argparse
import json
import os
import torch
from argparser import *
from os.path import join
import torch.utils.data as data
import torchvision.utils as vutils
from model.model import AttGAN
from data.dataloader import CelebA
from torchvision.utils import save_image
from utils.utils import find_model,Progressbar
from utils.utils import check_attribute_conflict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(args):
    
    """
    This function test the AttGAN model and generates sample images from test dataset
    and save in different forlder according attribute wise.For thirteen attributes it generate
    13 images for each input image.These images are used to evaluate our model.
    
    Params:
    
    --root_path                 : root path of your system as string.
    -- image_container_name     : Directory name where images store in root_path
    -- test_manifest_file_name  : test attributes file name
    -- weights_path             : directory name of weights 
    -- weights_name             : name of the weights as string
    -- num_test                 : num of test images use from test dataset - integer 
    
    Returns:
    This function returns None. But it saved the attribute wise generate image in seperate
    directory.
    
    """

    
    output_path = join(args.root_path,'generated_images')
    test_dataset = CelebA(args.root_path, args.test_manifest_file_name,
                               args.image_container_name,
                               args.img_size, args.attrs)

    os.makedirs(join(args.root_path,'generated_images'),
                exist_ok=True)
    
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, drop_last=False
    )
    
    print('Testing images:', len(test_dataset))
    
    attgan = AttGAN()
    attgan.load(join(args.weights_path, args.weights_name))
    progressbar = Progressbar()
    attgan.eval()
    
    for idx, (img_a, att_a) in enumerate(test_dataloader):
        if idx == 50:
            break
        img_a = img_a.to(device)
        att_a = att_a.to(device)
        att_a = att_a.type(torch.float)
        att_b_list = [att_a]
        for i in range(len(args.attrs)):
            tmp = att_a.clone()
            tmp[:, i] = 1 - tmp[:, i]
            tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
            att_b_list.append(tmp)
            
        s = test_dataset.images[idx]
        img_name = s[:-4] +'.jpg'
        print(img_name)
        os.makedirs(join(args.root_path,'generated_images'+"/real"),
                exist_ok=True)
        vutils.save_image(
                img_a, join(output_path+"/real", img_name),
                nrow=1, normalize=True, range=(-1., 1.)
        )
                
        with torch.no_grad():
            for i, att_b in enumerate(att_b_list):
                att_b_ = (att_b * 2 - 1) * args.thres_int
                if i > 0:
                    att_b_[..., i - 1] = att_b_[..., i - 1] * args.test_int / args.thres_int
                p = attgan.G(img_a, att_b_)
                
                os.makedirs(join(args.root_path,'generated_images'+'/attr-'+str(i)),
                exist_ok=True)
                vutils.save_image(
                p, join(output_path+'/attr-'+str(i), img_name),
                nrow=1, normalize=True, range=(-1., 1.)
            )


if __name__ == '__main__':
    args = test_parser()
    print(args)
    test(args)
