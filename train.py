import datetime
import json
import os
from os.path import join
import torch.utils.data as data
import torch
from argparser import *
import torchvision.utils as vutils
from model.model import AttGAN
from data.dataloader import CelebA
from utils.utils import Progressbar, add_scalar_dict
from utils.utils import check_attribute_conflict, find_model
from tensorboardX import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    """
    This function train the AttGAN model and validate.It generates sample images from valid dataset.
    
    Params:
    
    -- root_path                     : root path of your system as string.
    -- image_container_name          : Directory name where images store in root_path
    -- train_manifest_file_name      : train attributes file name
    -- validation_manifest_file_name : validation attributes file name
    -- epochs                        : integer,No of epoch to train model
    -- batch_size                    : integer, batch_size of dataset.
    -- save_interval                 : integer,it will save model after save_interval
    -- sample_interval               : integer,it will save generated sample images on valid dataset 
    -- num_test                      : num of test images use from test dataset - integer
    -- img_size                      : size of image 
    -- img_a                         : Input image contains tensor of image dimention
    -- att_a                         : Attribute holds the rael image
    -- att_b_list                    : list of attributes by fliping every attribute 
    
    Returns:
    None. It just save model checkpoint and sample generated images after sample interval
    
    """
    
    os.makedirs(join('output', args.experiment_name), exist_ok=True)
    os.makedirs(join('output', args.experiment_name, 'checkpoint'),
                exist_ok=True)
    os.makedirs(join('output', args.experiment_name, 'sample_training'),
                exist_ok=True)
    
    train_dataset = CelebA(args.root_path, args.train_manifest_file_path,
                           args.image_container_path,
                           args.img_size, args.attrs)
    valid_dataset = CelebA(args.root_path,
                           args.validation_manifest_file_path,
                           args.image_container_path,
                           args.img_size, args.attrs)

    train_dataloader = data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, drop_last=True
    )
    valid_dataloader = data.DataLoader(
        valid_dataset, batch_size=args.n_samples,
        shuffle=False, drop_last=False
    )
    trn_img = 'Training images:'
    val_img = 'Validating images:'
    print(trn_img, len(train_dataset), '/', val_img, len(valid_dataset))

    attgan = AttGAN()
    progressbar = Progressbar()
    writer = SummaryWriter(join('output', args.experiment_name, 'summary'))

    fixed_img_a, fixed_att_a = next(iter(valid_dataloader))
    fixed_img_a = fixed_img_a.to(device) 
    fixed_att_a = fixed_att_a.to(device)
    fixed_att_a = fixed_att_a.type(torch.float)
    sample_att_b_list = [fixed_att_a]
    for i in range(len(args.attrs)):
        tmp = fixed_att_a.clone()
        tmp[:, i] = 1 - tmp[:, i]
        tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
        sample_att_b_list.append(tmp)

    it = 0
    n_d = 1
    lr  = 0.0002
    thres_int = 0.5

    it_per_epoch = len(train_dataset) // args.batch_size
    for epoch in range(args.epochs):
        # train with base lr in the first 100 epochs
        # and half the lr in the last 100 epochs
        lr = lr / (10 ** (epoch // 100))
        attgan.set_lr(lr)
        writer.add_scalar('LR/learning_rate', lr, it+1)
        for img_a, att_a in progressbar(train_dataloader):
            attgan.train()
            img_a = img_a.to(device) 
            att_a = att_a.to(device) 
            idx = torch.randperm(len(att_a))
            att_b = att_a[idx].contiguous()

            att_a = att_a.type(torch.float)
            att_b = att_b.type(torch.float)

            att_a_ = (att_a * 2 - 1) * thres_int
            att_b_ = (att_b * 2 - 1) * thres_int
            input("hello world!")
            if (it+1) % (n_d+1) != 0:
                errD = attgan.trainD(img_a, att_a, att_a_, att_b, att_b_)
                add_scalar_dict(writer, errD, it+1, 'D')
            else:
                errG = attgan.trainG(img_a, att_a, att_a_, att_b, att_b_)
                add_scalar_dict(writer, errG, it+1, 'G')
                progressbar.say(epoch=epoch, iter=it+1, d_loss=errD['d_loss'],
                                g_loss=errG['g_loss'])
 
            if (it+1) % args.save_interval == 0:

                attgan.saveG(os.path.join(
                    'output', args.experiment_name, 'checkpoint',
                    'weights.{:d}.pth'.format(epoch)
                ))

            if (it+1) % args.sample_interval == 0:
                attgan.eval()
                with torch.no_grad():
                    samples = [fixed_img_a]
                    for i, att_b in enumerate(sample_att_b_list):
                        att_b_ = (att_b * 2 - 1) * args.thres_int
                        if i > 0:
                            att_b_[..., i - 1] = (att_b_[..., i - 1] *
                                                  args.test_int /
                                                  args.thres_int)
                        samples.append(attgan.G(fixed_img_a, att_b_))
                    samples = torch.cat(samples, dim=3)
                    writer.add_image('sample',
                                     vutils.make_grid(samples, nrow=1,
                                                      normalize=True,
                                                      range=(-1., 1.)), it+1)
                    vutils.save_image(samples, os.path.join(
                            output, args.experiment_name, 'sample_training',
                            'Epoch_({:d})_({:d}of{:d}).jpg'.format(
                                epoch, (it % it_per_epoch + 1), it_per_epoch)
                        ), nrow=1, normalize=True, range=(-1., 1.))
            it += 1


if __name__ == '__main__':
    
    args = train_parser()
    train(args)

