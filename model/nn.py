import torch.nn as nn


def add_normalization_1d(layers, fn, n_out):
    
    """
    add_normalization_1d
    
    Params: 
    -- layers        : holds all layers
    -- fn            : Normalized function 
    -- n_out         : Input for Normalized function
    """
    
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm1d(n_out))
    elif fn == 'instancenorm':
        layers.append(Unsqueeze(-1))
        layers.append(nn.InstanceNorm1d(n_out, affine=True))
        layers.append(Squeeze(-1))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers


def add_normalization_2d(layers, fn, n_out):
    
    """
    add_normalization_2d
    
    Params: 
    -- layers        : holds all layers
    -- fn            : Normalized function 
    -- n_out         : Input for Normalized function
    """
    
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm2d(n_out))
    elif fn == 'instancenorm':
        layers.append(nn.InstanceNorm2d(n_out, affine=True))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers


def add_activation(layers, fn):
    
    """
    add_activation
    
    Params: 
    -- layers        : holds all layers
    -- fn            : Activation function
    
    Return:
    Layer after passing through the activation function
    """
    
    if fn == 'none':
        pass
    elif fn == 'relu':
        layers.append(nn.ReLU())
    elif fn == 'lrelu':
        layers.append(nn.LeakyReLU())
    elif fn == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif fn == 'tanh':
        layers.append(nn.Tanh())
    else:
        raise Exception('Unsupported activation function: ' + str(fn))
    return layers


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class LinearBlock(nn.Module):
    
    """
    Fully connected portion 
    
    Params:
    -- n_in           : Input channel for linear block
    -- n_out          : Output channel for linear block
    -- norm_fn        : None
    -- acti_fn        : None
    
    Return:
    Fully connected layer result
    """
    def __init__(self, n_in, n_out, norm_fn='none', acti_fn='none'):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(n_in, n_out, bias=(norm_fn == 'none'))]
        layers = add_normalization_1d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Conv2dBlock(nn.Module):
    
    """
    Conv2dBlock for upsampling the images
    
    Param:
    -- n_in            : Number input channel for convulation layer
    -- n_out           : Number output channel for convulation layer
    -- kernel_size     : Number of karnel
    -- stride          : Value of stride
    -- padding         : Value of padding
    -- norm_fn         : Normalized function
    -- acti_fn         : Activation function
    
    Return:
    Upsample images from encoder portion
    """
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0,
                 norm_fn=None, acti_fn=None):
        super(Conv2dBlock, self).__init__()
        layers = [nn.Conv2d(n_in, n_out, kernel_size, stride=stride,
                            padding=padding, bias=(norm_fn == 'none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvTranspose2dBlock(nn.Module):
    """
    ConvTranspose2dBlock for downsampling the images
    
    Param:
    -- n_in            : Number input channel for convulation layer
    -- n_out           : Number output channel for convulation layer
    -- kernel_size     : Number of karnel
    -- stride          : Value of stride
    -- padding         : Value of padding
    -- norm_fn         : Normalized function
    -- acti_fn         : Activation function
    
    Return:
    downsampled images from decoder portion
    """
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0,
                 norm_fn=False, acti_fn=None):
        super(ConvTranspose2dBlock, self).__init__()
        layers = [nn.ConvTranspose2d(n_in, n_out, kernel_size, stride=stride,
                                     padding=padding,
                                     bias=(norm_fn == 'none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
