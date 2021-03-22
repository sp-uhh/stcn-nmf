import math
import torch
from torch import nn, optim


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, 
        dropout=0.2, activation=torch.tanh):
        super(ConvBlock, self).__init__()
        
        self.padding = nn.ConstantPad1d(((kernel_size - 1) * dilation, 0), 0)
        self.conv = nn.utils.weight_norm(nn.Conv1d(in_channels, 
            out_channels, kernel_size, dilation=dilation))
        self.activation = activation
        self.dropout = nn.Dropout(dropout)         
        self.init_weights()
        
    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)
        
    def forward(self, x):
        return self.dropout(self.activation(self.conv(self.padding(x))))


class ResidualBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, 
        dropout=0.2, activation=nn.ReLU()):
        super(ResidualBlockRes, self).__init__()
        
        self.resample = (nn.utils.weight_norm(nn.Conv1d(in_channels, 
            out_channels, 1)) if in_channels != out_channels else None)
        self.padding = nn.ConstantPad1d(((kernel_size - 1) * dilation, 0), 0)
        self.convolution = nn.utils.weight_norm(nn.Conv1d(out_channels, 
            out_channels, kernel_size, dilation=dilation))
        self.activation = activation
        self.dropout = nn.Dropout(dropout)         
        self.init_weights()
        
    def init_weights(self):
        self.convolution.weight.data.normal_(0, 0.01)
        if self.resample is not None:
            self.resample.weight.data.normal_(0, 0.01)
        
    def forward(self, x):
        x = x if self.resample is None else self.resample(x)
        y = self.dropout(self.activation(self.convolution(self.padding(x))))
        return self.activation(x + y)


class TCN(nn.Module):
    def __init__(self, channels, kernel_size=2, dropout=0.2, 
        activation=torch.tanh):
        super(TCN, self).__init__()
        
        self.channels = channels
        self.layers = nn.Sequential(*[ConvBlock(channels[i], channels[i+1],
            kernel_size, 2**i, dropout, activation) for i in range(
            len(channels) - 1)]) 
    
    def representations(self, x):
        # bottom-up
        d = [x]
        for i in range(len(self.channels)-1):
            d += [self.layers[i](d[-1])]
        return d[1:]
    
    def forward(self, x):
        return self.layers(x)
    

class LatentLayer(nn.Module):
    def __init__(self, tcn_dim, latent_dim_in, latent_dim_out, num_hid_layers):
        super(LatentLayer, self).__init__()
        
        self.num_hid_layers = num_hid_layers
        
        self.enc_in = nn.Linear(tcn_dim+latent_dim_in, tcn_dim)
        self.enc_hidden = nn.Sequential(*[nn.Linear(tcn_dim, tcn_dim) 
                                         for _ in range(num_hid_layers-1)])
        self.enc_out_1 = nn.Linear(tcn_dim, latent_dim_out)
        self.enc_out_2 = nn.Linear(tcn_dim, latent_dim_out)

    def forward(self, x):
        h = torch.transpose(x, 1, 2) # (N, D+Z, L) -> (N, L, D+Z)
        h = torch.tanh(self.enc_in(h))
        for i in range(self.num_hid_layers-1):
            h = torch.tanh(self.enc_hidden[i](h))  
        mu = torch.transpose(self.enc_out_1(h), 1, 2) # (N, L, D) -> (N, D, L)
        logvar = torch.transpose(self.enc_out_1(h), 1, 2) 
        return mu, logvar


class Encoder(nn.Module):
    def __init__(self, tcn_channels, latent_channels, concat_z, num_hid_layers):
        super(Encoder, self).__init__()
        
        self.layers = [LatentLayer(tcn_channels[i], latent_channels[i+1], 
            latent_channels[i], num_hid_layers) \
            for i in range(len(tcn_channels)-1)]  
        self.layers += [LatentLayer(tcn_channels[-1], 0, latent_channels[-1], 
            num_hid_layers)]
        self.layers = nn.ModuleList(self.layers) 

        self.min_logvar = math.log(0.001)
        self.max_logvar = math.log(5.0)
        self.concat_z = concat_z
                
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, d):
        # top-down
        mu_q_hat, logvar_q_hat = self.layers[-1](d[-1])   
        logvar_q = [logvar_q_hat]
        mu_q = [mu_q_hat]   
        z = [self.reparameterize(mu_q[0], logvar_q[0])]
        for i in reversed(range(len(self.layers)-1)):
            mu_q_hat, logvar_q_hat = self.layers[i](torch.cat((d[i], z[-1]), 
                dim=1))
            logvar_q += [logvar_q_hat]
            mu_q += [logvar_q_hat]         
            z += [self.reparameterize(mu_q[-1], logvar_q[-1])]
        z.reverse(); mu_q.reverse(); logvar_q.reverse() # [z_1, ..., z_L]
        if self.concat_z: 
            z = torch.cat([z[l] for l in range(len(z))], dim=1)
        else:
            z = z[0]
        return z, mu_q, logvar_q


class Decoder(nn.Module):
    def __init__(self, dec_channels):
        super(Decoder, self).__init__()
                
        self.dec_layers = nn.ModuleList([nn.Linear(dec_channels[i], 
            dec_channels[i+1]) for i in range(len(dec_channels) - 1)])

    def forward(self, h):
        h = torch.transpose(h, 1, 2) # (N, F, L) -> (N, L, F)
        for dec_layer in self.dec_layers[:-1]: 
            h = torch.tanh(dec_layer(h))  
        recon = torch.exp(self.dec_layers[-1](h)) 
        recon = torch.transpose(recon, 1, 2) # (N, L, F) -> (N, F, L)
        return recon


class DecoderTCN(nn.Module):
    def __init__(self, dec_channels, kernel_size=2, dropout=0.1, activation=torch.tanh):
        super(DecoderTCN, self).__init__()

        self.pad = [nn.ConstantPad1d(((kernel_size - 1) * 2**i, 0), 0) \
            for i in range(len(dec_channels) - 1)]

        self.layers = nn.Sequential(*[nn.Conv1d(dec_channels[i], dec_channels[i+1],
            kernel_size, dilation=2**i, padding_mode='reflect') for i in range(
            len(dec_channels) - 1)]) 

    def forward(self, h):
        for i, layer in enumerate(self.layers[:-1]): 
            h = self.pad[i](h)
            h = torch.tanh(layer(h))  
        h = self.pad[-1](h)
        recon = torch.exp(self.layers[-1](h)) 
        return recon
    

class STCN(nn.Module):
    def __init__(self, input_dim, tcn_channels, latent_channels, dec_channels,
                 concat_z=False, dropout=0.2, kernel_size=2):
        super(STCN, self).__init__()
        
        self.tcn = TCN([input_dim]+tcn_channels, kernel_size, dropout)   
        self.encoder = Encoder(tcn_channels, latent_channels, concat_z, 
            num_hid_layers=1)    
        self.decoder = Decoder(dec_channels)
        
    def encode(self, x):
        d = self.tcn.representations(x[None,:,:]) 
        z_q, mu_q, logvar_q = self.encoder(d)
        return z_q[0,:,:]
    
    def decode(self, z):
        x_hat = self.decoder(z[None,:,:])
        return x_hat[0,:,:]

    def forward(self, x):
        d = self.tcn.representations(x)
        z_q, mu_q, logvar_q = self.encoder(d)
        x_hat = self.decoder(z_q)
        return x_hat, mu_q, logvar_q
