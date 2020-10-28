import math
import torch
from torch import nn, optim


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, 
        dropout=0.2, activation=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        
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
        activation=nn.ReLU()):
        super(TCN, self).__init__()
        
        self.channels = channels
        self.layers = nn.Sequential(*[ResidualBlock(channels[i], channels[i+1],
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
    def __init__(self, tcn_dim, latent_dim_in, latent_dim_out, hidden_dim, 
        num_hidden_layers):
        super(LatentLayer, self).__init__()
        
        self.num_hidden_layers = num_hidden_layers
        
        self.enc_in = nn.Conv1d(tcn_dim+latent_dim_in, hidden_dim, 1)
        self.enc_hidden = nn.Sequential(*[nn.Conv1d(hidden_dim, hidden_dim, 1) 
                                         for _ in range(num_hidden_layers)])
        self.enc_out_1 = nn.Conv1d(hidden_dim, latent_dim_out, 1)
        self.enc_out_2 = nn.Conv1d(hidden_dim, latent_dim_out, 1)

    def forward(self, x):
        h = torch.tanh(self.enc_in(x))
        for i in range(self.num_hidden_layers):
            h = torch.tanh(self.enc_hidden[i](h))     
        return self.enc_out_1(h), self.enc_out_2(h)


class PriorModel(nn.Module):
    def __init__(self, tcn_channels, latent_channels, num_hidden_layers):
        super(PriorModel, self).__init__()
        
        self.layers = [LatentLayer(tcn_channels[i], latent_channels[i+1], 
            latent_channels[i], latent_channels[i], num_hidden_layers) \
            for i in range(len(tcn_channels)-1)]  
        self.layers += [LatentLayer(tcn_channels[-1], 0, latent_channels[-1], 
            latent_channels[-1], num_hidden_layers)]
        self.layers = nn.ModuleList(self.layers)

        self.min_logvar = math.log(0.001)
        self.max_logvar = math.log(5.0)
                
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
                
    def forward(self, d):
        # top-down
        mu_p, logvar_p = self.layers[-1](d[-1])
        mu_p = [mu_p]; logvar_p = [torch.clamp(logvar_p, self.min_logvar, 
            self.max_logvar)]
        z = [self.reparameterize(mu_p[-1], logvar_p[-1])]
        for i in reversed(range(len(self.layers)-1)):
            _mu_p, _logvar_p = self.layers[i](torch.cat((d[i], z[-1]), dim=1))
            z += [self.reparameterize(_mu_p, torch.clamp(_logvar_p, 
                self.min_logvar, self.max_logvar))]
            mu_p += [_mu_p]
            logvar_p += [torch.clamp(_logvar_p, self.min_logvar, 
                self.max_logvar)]
        z.reverse(); mu_p.reverse(); logvar_p.reverse()
        z_cat = torch.cat([z[l] for l in range(len(z))], dim=1)
        return z_cat, mu_p, logvar_p


class InferenceModel(nn.Module):
    def __init__(self, tcn_channels, latent_channels, num_hidden_layers):
        super(InferenceModel, self).__init__()
        
        self.layers = [LatentLayer(tcn_channels[i], latent_channels[i+1], 
            latent_channels[i], latent_channels[i], num_hidden_layers) \
            for i in range(len(tcn_channels)-1)]  
        self.layers += [LatentLayer(tcn_channels[-1], 0, latent_channels[-1], 
            latent_channels[-1], num_hidden_layers)]
        self.layers = nn.ModuleList(self.layers) 

        self.min_logvar = math.log(0.001)
        self.max_logvar = math.log(5.0)
                
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, d, mu_p, logvar_p):
        # top-down
        mu_q_hat, logvar_q_hat = self.layers[-1](d[-1])   
        logvar_q = [torch.clamp(2*(logvar_q_hat+logvar_p[-1]), self.min_logvar, 
            self.max_logvar)]
        mu_q = [torch.exp(logvar_q[0])*(mu_q_hat*torch.pow(torch.exp(
            logvar_q_hat), -2) + mu_p[-1]*torch.pow(torch.exp(
            logvar_p[-1]), -2))]   
        z = [self.reparameterize(mu_q[0], logvar_q[0])]
        for i in reversed(range(len(self.layers)-1)):
            mu_q_hat, logvar_q_hat = self.layers[i](torch.cat((d[i], z[-1]), 
                dim=1))
            logvar_q_hat = torch.clamp(logvar_q_hat, self.min_logvar, 
                self.max_logvar)
            logvar_q += [torch.clamp(2*(logvar_q_hat+logvar_p[i]), 
                self.min_logvar, self.max_logvar)]
            mu_q += [torch.exp(logvar_q[-1])*(mu_q_hat*torch.pow(torch.exp(
                logvar_q_hat), -2) + mu_p[i]*torch.pow(torch.exp(
                logvar_p[i]), -2))]         
            z += [self.reparameterize(mu_q[-1], logvar_q[-1])]
        z.reverse(); mu_q.reverse(); logvar_q.reverse()
        z_cat = torch.cat([z[l] for l in range(len(z))], dim=1)
        return z_cat, mu_q, logvar_q


class GenerativeModelTCN(TCN):
    def __init__(self, channels, kernel_size=2, dropout=0.2, 
        activation=nn.ReLU()):
        super(GenerativeModelTCN, self).__init__(channels, kernel_size, dropout, 
            activation)

    def forward(self, z):
        z_cat = torch.cat([z[l] for l in range(len(z))], dim=1)
        return self.layers(z_cat)
                
    
class GenerativeModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden_layers):
        super(GenerativeModel, self).__init__()
                
        self.num_hidden_layers = num_hidden_layers
        
        self.input = nn.Conv1d(input_dim, hidden_dim, 1)
        self.hidden = nn.Sequential(*[nn.Conv1d(hidden_dim, hidden_dim, 1) 
                                         for _ in range(num_hidden_layers)])
        self.out = nn.Conv1d(hidden_dim, output_dim, 1)

    def forward(self, z):
        # z_cat = torch.cat([z[l] for l in range(len(z))], dim=1)
        h = torch.tanh(self.input(z))
        for l in range(self.num_hidden_layers):
            h = torch.tanh(self.hidden[l](h)) 
        x_hat = torch.exp(self.out(h))
        return x_hat
    

class STCN(nn.Module):
    def __init__(self, input_dim, tcn_channels, latent_channels,
                 kernel_size=2, dropout=0.2):
        super(STCN, self).__init__()
        
        self.tcn = TCN([input_dim]+tcn_channels, kernel_size, dropout)   
        self.prior = PriorModel(tcn_channels, latent_channels, 
            num_hidden_layers=1)
        self.approx_posterior = InferenceModel(tcn_channels, latent_channels, 
            num_hidden_layers=1)    
        self.generative_model = GenerativeModel(input_dim=sum(latent_channels),
            output_dim=input_dim, hidden_dim=256, num_hidden_layers=2)

    def generate(self, x):
        d = self.tcn.representations(x) 
        d_shift = [(nn.functional.pad(d[i], pad=(1, 0))[:,:,:-1]) \
            for i in range(len(d))]   
        z_p, _, _ = self.prior(d_shift)
        x_hat = self.generative_model(z_p)          
        return x_hat

    def encode(self, x):
        d = self.tcn.representations(x[None,:,:]) 
        d_shift = [(nn.functional.pad(d[i], pad=(1, 0))[:,:,:-1]) \
            for i in range(len(d))]    
        z_p, mu_p, logvar_p = self.prior(d_shift)
        z_q, mu_q, logvar_q = self.approx_posterior(d, mu_p, logvar_p)
        return z_q[0,:,:]
    
    def decode(self, z):
        x_hat = self.generative_model(z[None,:,:])
        return x_hat[0,:,:]

    def forward(self, x):
        d = self.tcn.representations(x) 
        d_shift = [(nn.functional.pad(d[i], pad=(1, 0))[:,:,:-1]) \
            for i in range(len(d))]    
        z_p, mu_p, logvar_p = self.prior(d_shift)
        z_q, mu_q, logvar_q = self.approx_posterior(d, mu_p, logvar_p)
        x_hat = self.generative_model(z_q)
        return x_hat, z_p, z_q, mu_p, logvar_p, mu_q, logvar_q
    

def loss_function(x, recon_x, mu_p, logvar_p, mu_q, logvar_q, annealing): 
    # Kullback-Leibler divergence
    N, D, L =  logvar_q[-1].shape
    num_latent_layers = len(mu_p)
    loss_KL = 1/(2*N*L)*torch.sum(torch.pow(torch.exp(logvar_p[-1]), -1)
        *torch.exp(logvar_q[-1]) + torch.exp(logvar_p[-1])
        *torch.pow(mu_p[-1]-mu_q[-1], 2) + logvar_p[-1]-logvar_q[-1])-D/2 
    for i in range(num_latent_layers-1):
        var_p = torch.exp(logvar_p[i])
        var_q = torch.exp(logvar_q[i])
        N, D, L = var_p.shape
        loss_KL += 1/(2*N*L)*torch.sum(torch.pow(var_p, -1)*var_q 
            + var_p*torch.pow(mu_p[i]- mu_q[i], 2) + torch.log(var_p) 
            - torch.log(var_q))-D/2

    # Reconstruction loss
    N, F, L = x.shape  
    recon_loss = 1/(N*F*L)*torch.sum(torch.pow(torch.log(x+1e-8) 
        - torch.log(recon_x+1e-8), 2))
    
    return annealing*loss_KL + recon_loss
    

    