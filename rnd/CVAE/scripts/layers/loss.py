import torch
import numpy as np
import torch.nn.functional as F

def calculate_loss(reconstructed_x,x, mean, log_var): 
    RCL = F.mse_loss(reconstructed_x, x)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return  KLD + RCL
