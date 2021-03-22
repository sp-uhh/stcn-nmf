import os
import sys
import time
import glob
import torch
import pickle
import matplotlib.pyplot as plt 

from torch.utils.data import DataLoader
from dataset import Spectrogram, collate_fn
from stcn import STCN
from utils import count_parameters


############################ SETTINGS ##########################################

batch_size = 16
learning_rate = 1e-3
num_annealing = 10
epochs = 200
input_dim = 513

kernel_size = 2
tcn_channels = [128, 128]
latent_channels = [16, 4]
dec_channels = [16, 128, 128, 513]
concat_z = False
dropout = 0.1

verbose = True

########################## CONFIGURATION #######################################

# Time stemp for model file
time_stamp = time.localtime() 
time_string = time.strftime("%Y-%m-%d_%H:%M:%S", time_stamp)
print('stcn_{}'.format(time_string))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training and validation data
train_data = pickle.load(open('data/si_tr_s_split.p', 'rb'))
valid_data = pickle.load(open('data/si_dt_05_split.p', 'rb'))
    
train_dataset = Spectrogram(train_data)
valid_dataset = Spectrogram(valid_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
    collate_fn=collate_fn, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
    collate_fn=collate_fn, drop_last=True)

# Create model
model = STCN(input_dim, tcn_channels, latent_channels, dec_channels, concat_z,
    dropout, kernel_size).to(device)  

print('Parameters: {}'.format(count_parameters(model)))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

batch_loss_vec = []; train_loss_vec = []; valid_loss_vec = []

############################## TRAINING ########################################

def loss_function(x, recon_x, mu, logvar, annealing): 
    # Reconstruction loss
    _, _, L = x.shape   
    recon = 1/L*torch.sum(x/(recon_x) - torch.log(x/(recon_x)) - 1) 

    KL = 0
    # Kullback-Leibler divergence  
    for i in range(len(mu)):
        KL += -0.5/L * torch.sum(logvar[i] - mu[i].pow(2) - logvar[i].exp())

    return recon + annealing*KL, recon, KL

def train(epoch):
    model.train()
    train_loss = 0; batch_loss = 0; recon_loss = 0; KL_loss = 0
    annealing = min(1.0, float(epoch)/float(num_annealing)) 
    for batch_idx, data in enumerate(train_loader):
        x = data.to(device)
        optimizer.zero_grad()
        recon_x, mu_q, logvar_q = model(x)
        loss, recon, KL = loss_function(x, recon_x, mu_q, logvar_q, annealing)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        batch_loss_vec.append(loss.item()/batch_size)

        if verbose: 
            sys.stdout.write("\033[K") 
            print('\r', ' Epoch: {}   Batch {}/{}:  Loss: {:.0f}  Recon: {:.0f}  KL: {:.2f}'
                .format(epoch, batch_idx+1, len(train_loader), 
                loss.item()/batch_size, recon.item()/batch_size, 
                KL.item()/batch_size), 
                end="\r")
    return train_loss
      
    
def validate():
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(valid_loader):
            x = data.to(device)
            recon_x, mu_q, logvar_q = model(x)
            loss, recon, KL = loss_function(x, recon_x, mu_q, logvar_q, 1.0)
            valid_loss += loss.item()
    return valid_loss

# Start training
for epoch in range(1, epochs + 1):

    # Training step
    train_loss = train(epoch)

    # Validation step
    valid_loss = validate()

    # Append train and validation loss for plotting
    train_loss_vec.append(train_loss/len(train_loader.dataset))
    valid_loss_vec.append(valid_loss/len(valid_loader.dataset))

    # Print train and validation loss
    if verbose: sys.stdout.write("\033[K") 
    print('- Epoch: {}    Train loss: {:.0f}   Validation loss: {:.0f}'.format(
        epoch, train_loss/len(train_loader.dataset), 
        valid_loss/len(valid_loader.dataset)))

    # Plot loss curve
    plt.figure(0)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,4))
    ax1.plot(batch_loss_vec, label='batch_loss')
    ax1.set_xlabel('# batches')
    ax1.set_ylim(top=900)
    ax1.set_ylim(bottom=300)
    ax1.legend(loc='best')
    ax2.plot(train_loss_vec, label='train_loss')
    ax2.plot(valid_loss_vec, label='valid_loss')
    ax2.set_xlabel('epochs')
    ax2.legend(loc='best')
    f.savefig('loss/loss_{}.png'.format(time_string))
    plt.close()

    # Save best model
    if valid_loss_vec[-1] == min(valid_loss_vec):
        fileList = glob.glob('models/stcn_{}*.pt'.format(time_string))
        if fileList: os.remove(fileList[0])
        torch.save(model.state_dict(), 'models/stcn_{}_{:03d}_vloss_{:.0f}.pt'.format(
            time_string, epoch, valid_loss / len(valid_loader.dataset)))

    # Save model hyper-parameters
    if epoch == 1:
        with open('models/stcn_{}_param.pkl'.format(time_string), 'wb') as f:  
            pickle.dump([input_dim, tcn_channels, latent_channels, dec_channels, 
                concat_z, dropout, kernel_size], f)
