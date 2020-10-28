import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, in_out_dim, hid_dim, latent_dim , num_hid_layers):
        super(VAE, self).__init__()

        self.num_hid_layers = num_hid_layers
        self.in_out_dim = in_out_dim
        
        # encoder  
        self.enc_in = nn.Linear(in_out_dim, hid_dim)
        self.enc_hidden = nn.ModuleList([nn.Linear(hid_dim, hid_dim) 
            for _ in range(num_hid_layers - 1)])
        self.mu = nn.Linear(hid_dim, latent_dim)
        self.logvar = nn.Linear(hid_dim, latent_dim)
        
        # decoder
        self.dec_in = nn.Linear(latent_dim, hid_dim)
        self.dec_hidden = nn.ModuleList([nn.Linear(hid_dim, hid_dim) 
            for _ in range(num_hid_layers - 1)])
        self.dec_out = nn.Linear(hid_dim, in_out_dim)


    def encode(self, x):
        h = torch.tanh(self.enc_in(x.T))
        for i in range(self.num_hid_layers - 1):
            h = torch.tanh(self.enc_hidden[i](h))  
        mu = self.mu(h)
        logvar = self.logvar(h) 
        z = self.reparameterize(mu, logvar)   
        return z.T
    
    
    def decode(self, z):
        h = torch.tanh(self.dec_in(z.T))
        for i in range(self.num_hid_layers - 1):
            h = torch.tanh(self.dec_hidden[i](h))
        return torch.exp(self.dec_out(h)).T 


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def forward(self, x):
        h = torch.tanh(self.enc_in(x))
        for i in range(self.num_hid_layers - 1):
            h = torch.tanh(self.enc_hidden[i](h))  
        mu = self.mu(h)
        logvar = self.logvar(h) 
        z = self.reparameterize(mu, logvar)  
        h = torch.tanh(self.dec_in(z))
        for i in range(self.num_hid_layers - 1):
            h = torch.tanh(self.dec_hidden[i](h))
        recon = torch.exp(self.dec_out(h))   
        return recon, mu, logvar


def loss_function(x, recon_x, mu, logvar): 
    recon = torch.sum(x/recon_x - torch.log(x/recon_x) - 1) 
    KL = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp())
    return recon + KL
