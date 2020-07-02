from os.path import join
import os
import torch
import cv2
import multiprocessing
import numpy as np
import glob
import pathlib
import argparse
import pandas as pd
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
import torch.utils.data as data
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.models import inception_v3
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image

from inception import InceptionV3

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x
    
    
def inception_score(imgs, cuda=False, batch_size=10, resize=True, splits=1):
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device,\
                   so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False)\
                               .type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):

        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores)

def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]


def get_activations(files, model, batch_size=50, dims=2048,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    pred_arr = np.empty((len(files), dims))

    for i in tqdm(range(0, len(files), batch_size)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i
        end = i + batch_size

        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])

        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50,
                                    dims=2048, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda)

    return m, s


def calculate_fid_given_paths(paths, batch_size, cuda, dims):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size,
                                         dims, cuda)
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size,
                                         dims, cuda)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

def parser(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-path', type=str, help='root path type here',
                        default='/home/incentive/Desktop/FINAL Project/ojt-project/dataset/generated_images/')
    return parser.parse_args(args)

if __name__=="__main__":
    
    args = parser()
    print(args)
    root_path = args.root_path
    
    attributes= [ 'Reconstruction', 'Bald', 'Bangs', 'Black_Hair',
    'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male',
    'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']


    result = open(os.path.join(root_path, "evaluation.txt"), "w+")

    heading = "Attribute wise evaluation score of our AttGAN model are given bellow: \n\n"# 12 space
    result.write(heading)
    result.write("===============================================================================\n\n\n")
    result.write("< Attributes, Inception & FID Score  > \n\n")
    result.close()
    
    
    IS_score ,FID_score = [],[]
    for i in range(14):
        fid_path = [root_path+'real',root_path+'attr-'+str(i)]
        FID = calculate_fid_given_paths(fid_path,10,False,2048)                                 

        name_list = os.listdir(root_path+'attr-'+str(i))
        images = [cv2.imread(root_path+'attr-'+str(i)+'/'+k).transpose(2,1,0) for k in name_list]

        IS = inception_score(images, cuda=False, batch_size=10, resize=False, splits=1)
        line = "{}: ( {:.2f},{:.2f} ) ".format(attributes[i],IS,FID)
        f = open(os.path.join(root_path, "evaluation.txt"), "a")
        print(line)
        FID_score.append(FID)
        IS_score.append(IS)
        f.write(line+'\n\n')
        f.close()  
    # plot FID histogram    
    bars = ('Reconstruction', 'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
            'Bushy_Eyebrows', 'Eyeglasses',   'Male', 'Mouth_Slightly_Open',
            'Mustache', 'No_Beard', 'Pale_Skin', 'Young')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, IS_score, color=(0.2, 0.4, 0.6, 0.6))
    plt.xticks(y_pos, bars, color='black', rotation=45, fontweight='bold', fontsize='8', horizontalalignment='right')
    plt.tick_params(labelbottom='off')
    plt.savefig(root_path+'IS.png')
    # plot IS histogram
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, FID_score, color=(0.2, 0.4, 0.6, 0.6))
    plt.xticks(y_pos, bars, color='black', rotation=45, fontweight='bold', fontsize='8', horizontalalignment='right')
    plt.tick_params(labelbottom='off')
    plt.savefig(root_path+'FID.png')