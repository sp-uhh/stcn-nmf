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


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, 
        dropout=0.2, activation=torch.tanh):
        super(ResBlock, self).__init__()
        
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
    def __init__(self, channels, kernel_size=2, res_net=True, dropout=0.2, 
        activation=torch.tanh, act_output_layer=torch.exp):
        super(TCN, self).__init__()
        
        self.channels = channels

        if res_net:
            conv_blocks = [ConvBlock(channels[i], channels[i+1],
                kernel_size, 2**i, dropout, activation) for i in range(
                len(channels) - 2)]
            conv_blocks += [ConvBlock(channels[-2], channels[-1],
                kernel_size, 2**(len(channels) - 2), dropout, act_output_layer)] 
            self.layers = nn.Sequential(*conv_blocks) 
        else:
            res_blocks = [ConvBlock(channels[i], channels[i+1],
                kernel_size, 2**i, dropout, activation) for i in range(
                len(channels) - 2)]
            res_blocks += [ConvBlock(channels[-2], channels[-1],
                kernel_size, 2**(len(channels) - 2), dropout, act_output_layer)] 
            self.layers = nn.Sequential(*res_blocks) 
    
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
        
        self.enc = nn.Sequential(*[nn.Linear(tcn_dim, tcn_dim) 
                                    for _ in range(num_hid_layers)])
        self.enc_out_1 = nn.Linear(tcn_dim + latent_dim_in, latent_dim_out)
        self.enc_out_2 = nn.Linear(tcn_dim + latent_dim_in, latent_dim_out)

    def forward(self, d, z):
        # Conversion from Conv1D to Linear (N, D, L) -> (N, L, D)
        d = torch.transpose(d, 1, 2) 
        for i in range(self.num_hid_layers):
            d = torch.tanh(self.enc[i](d))  

        # Concatenate d and z
        d_z_cat = torch.cat((d, z), dim=2)
        mu = self.enc_out_1(d_z_cat)
        logvar = self.enc_out_1(d_z_cat)
        return mu, logvar


class Posterior(nn.Module):
    def __init__(self, tcn_channels, latent_channels, num_enc_layers, concat_z,
        activation):
        super(Posterior, self).__init__()
        
        self.layers = [LatentLayer(tcn_channels[i+1], latent_channels[i+1], 
            latent_channels[i], num_enc_layers) \
            for i in range(len(tcn_channels)-2)]  
        self.layers += [LatentLayer(tcn_channels[-1], 0, latent_channels[-1], 
            num_enc_layers)]
        self.layers = nn.ModuleList(self.layers) 

        self.min_logvar = math.log(0.1)
        self.max_logvar = math.log(5.0)
        self.concat_z = concat_z
                
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, d):
        z = torch.zeros((d[0].shape[0], d[0].shape[2], 0), device="cuda")
        _mu_p, _logvar_p = self.layers[-1](d[-1], z)   
        mu_p = [_mu_p]   
        logvar_p = [torch.clamp(_logvar_p, self.min_logvar, self.max_logvar)]
        z = [self.reparameterize(mu_p[0], logvar_p[0])]
        for i in reversed(range(len(self.layers)-1)):
            _mu_p, _logvar_p = self.layers[i](d[i], z[-1])
            logvar_p += [torch.clamp(_logvar_p, self.min_logvar, 
                self.max_logvar)]
            mu_p += [_mu_p]         
            z += [self.reparameterize(mu_p[-1], logvar_p[-1])]
        z.reverse(); mu_p.reverse(); logvar_p.reverse() 
        # Concat z and convert from Linear to Conv1D (N, L, Z) -> (N, Z, L)  
        if self.concat_z: 
            z = torch.cat([torch.transpose(z[l], 1, 2) for l in range(len(z))], 
                dim=1)
        else:
            z = torch.transpose(z[0], 1, 2)
        return z, mu_p, logvar_p


class Prior(nn.Module):
    def __init__(self, tcn_channels, latent_channels, num_enc_layers, concat_z,
        activation):
        super(Prior, self).__init__()
        
        self.layers = [LatentLayer(tcn_channels[i+1], latent_channels[i+1], 
            latent_channels[i], num_enc_layers) \
            for i in range(len(tcn_channels)-2)]  
        self.layers += [LatentLayer(tcn_channels[-1], 0, latent_channels[-1], 
            num_enc_layers)]
        self.layers = nn.ModuleList(self.layers) 

        self.min_logvar = math.log(0.1)
        self.max_logvar = math.log(5.0)
        self.concat_z = concat_z


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, d, mu_p, logvar_p):
        # Input TCN output and dummy latent variable at topmost layer
        z = torch.zeros((d[0].shape[0], d[0].shape[2], 0), device="cuda")
        mu_q_hat, logvar_q_hat = self.layers[-1](d[-1], z)  

        # Clamp variance 
        logvar_q_hat = torch.clamp(logvar_q_hat, self.min_logvar, 
            self.max_logvar)

        # Precision-weighting
        _logvar_q = torch.log(1/(torch.exp(logvar_q_hat)**-1 
                                   + torch.exp(logvar_p[-1])**-1))

        # Clamp variance 
        logvar_q = [torch.clamp(_logvar_q, self.min_logvar, self.max_logvar)]

        # Precision-weighted arithmetric mean
        mu_q = [_logvar_q.exp()*(mu_q_hat*(logvar_q_hat.exp()**-1)
                  + mu_p[-1]*(logvar_p[-1].exp()**-1))]

        # Sample from approximate posterior    
        z = [self.reparameterize(mu_q[0], logvar_q[0])]

        # Top-down inference pass
        for i in reversed(range(len(self.layers)-1)):
            # Input TCN output and sampled latent variable
            mu_q_hat, logvar_q_hat = self.layers[i](d[i], z[-1])

            # Clamp variance  
            logvar_q_hat = torch.clamp(logvar_q_hat, self.min_logvar, 
                self.max_logvar)
            
            # Precision-weighting
            _logvar_q = torch.log(1/(logvar_q_hat.exp()**-1 
                                        + logvar_p[i].exp()**-1))

            # Clamp variance                             
            logvar_q += [torch.clamp(_logvar_q, self.min_logvar, 
                self.max_logvar)]

            # Precision-weighted arithmetric mean
            mu_q += [_logvar_q.exp()*(mu_q_hat*(logvar_q_hat.exp()**-1)
                    + mu_p[i]*(logvar_p[i].exp()**-1))]
     
            # Sample from approximate posterior  
            z += [self.reparameterize(mu_q[-1], logvar_q[-1])]

        # Reverse lists to get the right order (bottom-up)
        z.reverse(); mu_q.reverse(); logvar_q.reverse()

        # Concat z and convert from Linear to Conv1D (N, L, Z) -> (N, Z, L)  
        if self.concat_z: 
            z = torch.cat([torch.transpose(z[l], 1, 2) for l in range(len(z))], 
                dim=1)
        else:
            z = torch.transpose(z[0], 1, 2)
        return z, mu_q, logvar_q


class Decoder(nn.Module):
    def __init__(self, dec_channels, kernel_size=1, res_net=False, dropout=0.0, 
        activation=torch.tanh, activation_last_layer=torch.exp):
        super(Decoder, self).__init__()

        self.tcn = TCN(dec_channels, kernel_size, res_net, dropout, activation, 
            activation_last_layer)

    def forward(self, x):
        recon = self.tcn(x)
        return recon


class STCN(nn.Module):
    def __init__(self, tcn_channels, tcn_kernel, tcn_res, concat_z, 
        num_enc_layers, latent_channels, dec_channels, dec_kernal, dropout, 
        activation):
        super(STCN, self).__init__()
        
        self.tcn = TCN(tcn_channels, tcn_kernel, tcn_res, dropout, activation, 
            activation)  
        self.posterior = Posterior(tcn_channels, latent_channels, num_enc_layers, 
            concat_z, activation)
        self.prior = Prior(tcn_channels, latent_channels, num_enc_layers, 
            concat_z, activation)
        self.decoder = Decoder(dec_channels, dec_kernal, False, 0.0, torch.tanh,
            torch.exp)
        
    def encode(self, x):
        d = self.tcn.representations(x[None,:,:]) 
        d_shift = [(nn.functional.pad(d[i], pad=(0, 1))[:,:,1:]) \
            for i in range(len(d))]    
        z_q, _, _ = self.posterior(d)
        return z_q[0,:,:]
    
    def decode(self, z):
        recon_S = self.decoder(z[None,:,:])
        return recon_S[0,:,:]

    def forward(self, x):
        d = self.tcn.representations(x)
        d_shift = [(nn.functional.pad(d[i], pad=(0, 1))[:,:,1:]) \
            for i in range(len(d))]  
        z_q, mu_q, logvar_q = self.posterior(d)
        _, mu_p, logvar_p = self.prior(d_shift, mu_q, logvar_q)
        recon_S = self.decoder(z_q)
        return recon_S, mu_p, logvar_p, mu_q, logvar_q
