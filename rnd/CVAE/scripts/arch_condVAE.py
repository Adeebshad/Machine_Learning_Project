import torch
from torch.nn import Linear, ReLU,Tanh, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF


INPUT_DIM = 64*64*3    # size of each input
LATENT_DIM = 256         # latent vector dimension
N_CLASSES = 73   


import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, n_classes):
        super().__init__()
        
        self.cnn_layers = Sequential(
            
            # 3x 64 x 64 => 64 x 32 x 32
            Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            ReLU(inplace=True), 
            
            # 32 x 32 --> 16 x 16
            Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            BatchNorm2d(128),
            ReLU(inplace=True),
            
            # 16 x 16 --> 8 x 8
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            
            # 8 x 8 --> 4 x 4
            Conv2d(256, 256 ,kernel_size=3, stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            
            # 4 x 4 --> 1 x 1
            Conv2d(256, 1024 ,kernel_size=4, stride=1, padding=0),
            
            ReLU(inplace=True),
        )

        self.linear = nn.Linear(1024, 1024)
        self.drop_layer = nn.Dropout(0.5)
        
        self.linearzim = nn.Linear(1097,512)
        
        self.mu = nn.Linear(512, latent_dim)
        self.var = nn.Linear(512, latent_dim)

    def forward(self, x,y):
        
        cnn_x = self.cnn_layers(x)
        cnn_x = cnn_x.view(y.shape[0],-1)
        
        z = F.relu(self.linear(cnn_x))
        z = self.drop_layer(z)
        
        zim = torch.cat((z, y), dim=1)
        hidden = F.relu(self.linearzim(zim))
       
        mean = self.mu(hidden)
        log_var = self.var(hidden)
        
        return mean, log_var


    
class Decoder(nn.Module):
   
    def __init__(self, latent_dim, output_dim, n_classes):
       
        super().__init__()
        
        
        self.linearz = nn.Linear(329, 256)
        self.linearz2 = nn.Linear(256, 1024*1*1)
        
        self.cnn_layers = Sequential(
            
            #  1 x 1 --> 4 x 4 
            nn.ConvTranspose2d(1024, 256 ,kernel_size=4, stride=1, padding=0),
            ReLU(inplace=True),

            #   4 x 4 --> 8 x 8
            nn.ConvTranspose2d(256, 256 ,kernel_size=4, stride=2, padding=1),
            ReLU(inplace=True),

            #   8 x 8 -->  16 x 16
            nn.ConvTranspose2d(256,128, kernel_size=4, stride=2, padding=1),
            ReLU(inplace=True),

            #  16 x 16  --> 32 x 32
            nn.ConvTranspose2d(128,64, kernel_size=4, stride=2, padding=1),
            ReLU(inplace=True),

#           64 x 32 x 32 => 3x 64 x 64
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()   
        )

        self.linear_generated_x = nn.Linear(3*64*64, 3*64*64)
    
    def forward(self, z,y):
       
        z = torch.cat((z, y), dim=1)
        z = F.relu(self.linearz(z))
        z = F.relu(self.linearz2(z))
        z = z.view((-1,1024,1,1))
        
        generated_x = self.cnn_layers(z)
        generated_x=generated_x.view(-1, 3*64*64)
        generated_x = F.relu(self.linear_generated_x(generated_x))
        generated_x = generated_x.view((-1, 3,64,64))
        
        return generated_x



    
    
    

class CVAE(nn.Module):
    
    def __init__(self, input_dim, latent_dim, n_classes):  
        super().__init__()

        self.encoder = Encoder(input_dim,  latent_dim, n_classes)
        self.decoder = Decoder(latent_dim, input_dim, n_classes)

    def forward(self, x, y):

        z_mu, z_var = self.encoder(x,y)
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)
        generated_x = self.decoder(x_sample,y)
        return generated_x, z_mu, z_var
