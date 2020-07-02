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
from utils.utils import find_model,Progressbar
from utils.utils import check_attribute_conflict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(args):
    """
    This function test the AttGAN model and generates sample images from test dataset.
    It generate 13 image for each input image according attribute.
    
    Params:
    
    -- root_path                : root path of your system as string.
    -- image_container_name     : Directory name where images store in root_path
    -- test_manifest_file_name  : test attributes file name
    -- weights_path             : directory name of weights 
    -- thres_int                : float, it will use to normalize attributes value in specific threshold value
    -- weights_name             : name of the weights as string
    -- num_test                 : num of test images use from test dataset - integer
    -- img_size                 : size of image 
    -- img_a                    : Input image contains tensor of image dimention
    -- att_a                    : Attribute holds the rael image
    -- att_b_list               : list of attributes by fliping every attribute 
        
    Returns:
    None. It will generate attributes wise images from test dataset 
    
    """
    
    output_path = join('output','sample_testing')
    test_dataset = CelebA(args.root_path, args.test_manifest_file_path,
                               args.image_container_path,
                               args.img_size, args.attrs)

    os.makedirs(join('output','sample_testing'),
                exist_ok=True)
    
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, drop_last=False
    )
    
    print('Testing images:', len(test_dataset))
    

    attgan = AttGAN()
    attgan.load(join(args.root_path,args.weights_path))
    progressbar = Progressbar()
    attgan.eval()
    for idx, (img_a, att_a) in enumerate(test_dataloader):
        
        img_a = img_a.to(device)
        att_a = att_a.to(device)
        att_a = att_a.type(torch.float)
        att_b_list = [att_a]
        for i in range(len(args.attrs)):
            tmp = att_a.clone()
            tmp[:, i] = 1 - tmp[:, i]
            tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
            att_b_list.append(tmp)
        with torch.no_grad():
            samples = [img_a]
            for i, att_b in enumerate(att_b_list):
                att_b_ = (att_b * 2 - 1) * args.thres_int
                if i > 0:
                    att_b_[..., i - 1] = (att_b_[..., i - 1] *
                                          args.test_int / args.thres_int)
                samples.append(attgan.G(img_a, att_b_))
            samples = torch.cat(samples, dim=3)
            
            out_file = '{:06d}.jpg'.format(idx + 182638)
            vutils.save_image(
                samples, join(output_path, out_file),
                nrow=1, normalize=True, range=(-1., 1.)
            )
            print('{:s} done!'.format(out_file))


if __name__ == '__main__':
    args = test_parser()
#     print(args)
    test(args)
