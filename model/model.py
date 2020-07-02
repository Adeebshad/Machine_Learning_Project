import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from model.nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock
from torchsummary import summary

MAX_DIM = 64 * 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    """
    The Generator of AttGAN is designed here
    
    Params: 
    
    -- enc_dim             : enc_dim is used to keep the dimension up to 1024    
    -- dec_dim             : dec_dim is used to keep the dimension up to 1024
    -- enc_layers          : enc_layers is decleared for the encoder layer size
    -- dec_layers          : dec_layers is decleared for the encoder layer size
    -- enc_norm_fn         : enc_norm_fn is the normalization function assigned as 'batchnorm'
    -- dec_norm_fn         : enc_norm_fn is the normalization function assigned as 'batchnorm'
    -- enc_acti_fn         : Activation function of encoder lrelu
    -- dec_acti_fn         : Activation function of encoder relu
    -- n_attrs             : Number of attribute 
    -- shortcut_layers     : Adding some extra channel
    -- inject_layers       : Adding some more attribute channel
    -- img_size            : Image to be execute in the system
    """
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm',
                 enc_acti_fn='lrelu', dec_dim=64, dec_layers=5,
                 dec_norm_fn='batchnorm', dec_acti_fn='relu',
                 n_attrs=13, shortcut_layers=1, inject_layers=0, img_size=384):
        super(Generator, self).__init__()
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2**enc_layers

        layers = []
        n_in = 3
        for i in range(enc_layers):
            n_out = min(enc_dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn,
                acti_fn=enc_acti_fn
            )]
            n_in = n_out
        self.enc_layers = nn.ModuleList(layers)

        layers = []
        n_in = n_in + n_attrs  # 1024 + 13
        for i in range(dec_layers):
            if i < dec_layers - 1:
                n_out = min(dec_dim * 2**(dec_layers-i-1), MAX_DIM)
                layers += [ConvTranspose2dBlock(
                    n_in, n_out, (4, 4), stride=2, padding=1,
                    norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                )]
                n_in = n_out
                n_in = n_in + n_in//2 if self.shortcut_layers > i else n_in
                n_in = n_in + n_attrs if self.inject_layers > i else n_in
            else:
                layers += [ConvTranspose2dBlock(
                    n_in, 3, (4, 4), stride=2, padding=1, norm_fn='none',
                    acti_fn='tanh'
                )]
        self.dec_layers = nn.ModuleList(layers)

    def encode(self, x):
        
        """
        encoder function takes image input.
        
        Params:
        -- x             : Image input parameter
        
        Return:
        ecoder return a upsampling data to decoder
        
        """
        z = x
        zs = []
        for layer in self.enc_layers:
            z = layer(z)
            zs.append(z)
        return zs

    def decode(self, zs, a):
        
        """
        decode function complete the downsampling an generate image with respective image size and channel
        
        Params:
         
        -- a_tile        : Resized the imaged into tensor
        -- z             : Variable concatenate with atribute
        
        Return:
        An generated image in shape 3 * 384 * 384 
        """
        a_tile = a.view(a.size(0), -1, 1, 1) \
                  .repeat(1, 1, self.f_size, self.f_size)
        z = torch.cat([zs[-1], a_tile], dim=1)
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
            if self.shortcut_layers > i:  # Concat 1024 with 512
                z = torch.cat([z, zs[len(self.dec_layers) - 2 - i]], dim=1)
            if self.inject_layers > i:
                a_tile = a.view(a.size(0), -1, 1, 1) \
                          .repeat(1, 1, self.f_size * 2**(i+1),
                                  self.f_size * 2**(i+1))
                z = torch.cat([z, a_tile], dim=1)
        return z

    def forward(self, x, a=None, mode='enc-dec'):
        if mode == 'enc-dec':
            assert a is not None, 'No given attribute.'
            return self.decode(self.encode(x), a)
        if mode == 'enc':
            return self.encode(x)
        if mode == 'dec':
            assert a is not None, 'No given attribute.'
            return self.decode(x, a)
        raise Exception('Unrecognized mode: ' + mode)


class Discriminators(nn.Module):
    
    """
    The Discriminators of AttGAN is designed here
    
    Params: 
    
    -- dim               : dim is used to keep the dimension up to 1024    
    -- fc_dim            : fc_dim dimension is 1024
    -- n_layers          : n_layers is decleared 5
    -- acti_fn           : acti_fn function is lrelu
    -- fc_norm_fn        : Normalization function assigned as 'instancenorm'
    -- fc_acti_fn        : Activation function of encoder lrelu
    -- n_layers          : Number odf layers
    -- img_size          : Image to be execute in the system
    
    Return:
    Adversial Fully Connected opration result
    Attribute Classification Constrain result
    """
    def __init__(self, dim=64, norm_fn='instancenorm', acti_fn='lrelu',
                 fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu',
                 n_layers=5, img_size=384):
        super(Discriminators, self).__init__()
        self.f_size = img_size // 2**n_layers
        layers = []
        n_in = 3
        for i in range(n_layers):
            n_out = min(dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn,
                acti_fn=acti_fn
            )]
            n_in = n_out
        self.conv = nn.Sequential(*layers)
        self.fc_adv = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn,
                        fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none')
        )
        self.fc_cls = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn,
                        fc_acti_fn),
            LinearBlock(fc_dim, 13, 'none', 'none')
        )

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_adv(h), self.fc_cls(h)


class AttGAN():
    def __init__(self):
        """
        AttGAN calls Generator and Discriminators
        
        Params:
        -- lambda_1             : Loss calculation constant 100.0
        -- lambda_2             : Loss calculation constant 10.0
        -- lambda_3             : Loss calculation constant 1.0
        -- lambda_gp            : Loss calculation constant 10.0
        -- mode                 : Mode is defined with encoder/ decoder
        -- lr                   : Learning Rate 0.0002
        -- betas                : Betas values 0.5,0.999
        """
        
        self.lambda_1 = 100.0 	
        self.lambda_2 = 10.0 	
        self.lambda_3 = 1.0	
        self.lambda_gp = 10.0	
        self.mode = 'enc'
        self.img_size=384
        self.n_attrs=13
        self.G = Generator()
        self.G.train()
        
        self.G.to(device)
        k='cuda'
        if device=='cpu':
            k='cpu'
        summary(self.G, [(3, self.img_size, self.img_size),
                (self.n_attrs, 1, 1)], batch_size=4,
                device=k)

        self.D = Discriminators()
        self.D.train()
        
        self.D.to(device)
        summary(self.D, [(3, self.img_size, self.img_size)], batch_size=4,
                device=k)

        
        
        self.lr = 0.0002
        self.betas =(0.5,0.999)
        self.optim_G = optim.Adam(self.G.parameters(), lr=self.lr,
                                  betas=self.betas)
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr ,
                                  betas=self.betas)

    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_D.param_groups:
            g['lr'] = lr

    def trainG(self, img_a, att_a, att_a_, att_b, att_b_):
        
        """
        Generator Train
        
        Params:
        -- img_a        : Input image contains img_a
        -- att_a        : Attribute holds the rael image
        -- att_a_       : Costomized real attribute
        -- att_b        : Target attribute 
        -- att_b_       : Customized Target attribute 
        -- gf_loss      : Fake image loss
        -- gc_loss      : Attribute constraint classification loss
        -- gr_loss      : Reconstructed image loss
        -- g_loss       : Total image loss
        
        Return:
        Generator protion all loss result
        """
        for p in self.D.parameters():
            p.requires_grad = False

        zs_a = self.G(img_a, mode='enc')
        img_fake = self.G(zs_a, att_b_, mode='dec')
        img_recon = self.G(zs_a, att_a_, mode='dec')
        d_fake, dc_fake = self.D(img_fake)

        gf_loss = -d_fake.mean()
        gc_loss = F.binary_cross_entropy_with_logits(dc_fake, att_b)
        gr_loss = F.l1_loss(img_recon, img_a)
        g_loss = gf_loss + self.lambda_2 * gc_loss + self.lambda_1 * gr_loss

        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()

        errG = {
            'g_loss': g_loss.item(), 'gf_loss': gf_loss.item(),
            'gc_loss': gc_loss.item(), 'gr_loss': gr_loss.item()
        }
        return errG

    def trainD(self, img_a, att_a, att_a_, att_b, att_b_):
        
        """
        Discriminators Train
        
        Params:
        -- img_a        : Input image contains img_a
        -- att_a        : Attribute holds the rael image
        -- att_a_       : Costomized real attribute
        -- att_b        : Target attribute 
        -- att_b_       : Customized Target attribute 
        
        Return:
        Discriminators protion all loss result
        """
        for p in self.D.parameters():
            p.requires_grad = True
        img_fake = self.G(img_a, att_b_).detach()
        d_real, dc_real = self.D(img_a)
        d_fake, dc_fake = self.D(img_fake)

        def gradient_penalty(f, real, fake=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1)
                alpha = alpha.to(device) 
                inter = a + alpha * (b - a)
                return inter
            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = autograd.grad(
                outputs=pred, inputs=x,
                grad_outputs=torch.ones_like(pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp
        wd = d_real.mean() - d_fake.mean()
        df_loss = -wd
        df_gp = gradient_penalty(self.D, img_a, img_fake)
        dc_loss = F.binary_cross_entropy_with_logits(dc_real, att_a)
        d_loss = df_loss + self.lambda_gp * df_gp + self.lambda_3 * dc_loss

        self.optim_D.zero_grad()
        d_loss.backward()
        self.optim_D.step()

        errD = {
            'd_loss': d_loss.item(), 'df_loss': df_loss.item(),
            'df_gp': df_gp.item(), 'dc_loss': dc_loss.item()
        }
        return errD

    def train(self):
        self.G.train()
        self.D.train()

    def eval(self):
        self.G.eval()
        self.D.eval()

    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict()
        }
        torch.save(states, path)

    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
        if 'optim_D' in states:
            self.optim_D.load_state_dict(states['optim_D'])

    def saveG(self, path):
        states = {
            'G': self.G.state_dict()
        }
        torch.save(states, path)

