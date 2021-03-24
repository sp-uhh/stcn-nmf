import os
import sys
import time
import glob
import torch
import pickle
import matplotlib.pyplot as plt 

from torch.utils.data import DataLoader
from dataset import CleanSpeech
from stcn import STCN
from utils import count_parameters


############################ SETTINGS ##########################################

batch_size = 16
learning_rate = 1e-3
num_annealing = 5
epochs = 200
min_spec_energy = 1e-11
standard_normal_prior = True

tcn_channels = [513, 128, 128]
tcn_kernel = 2
tcn_res = False
concat_z = False
num_enc_layers = 1
latent_channels = [16, 16]
dec_channels = [16, 128, 128, 513]
dec_kernal = 1
dropout = 0.2
activation = torch.tanh

# Print batch loss every [batch_verbose] batch 
verbose = True
batch_verbose = 50

# For retraining an existing model 
retrain = False
stcn_path = ''
start_epoch = 1


########################## CONFIGURATION #######################################

# Time stemp for model file
if retrain:
    time_string = stcn_path[12:31]
else:
    time_stamp = time.localtime() 
    time_string = time.strftime("%Y-%m-%d_%H:%M:%S", time_stamp)
    print('stcn_{}'.format(time_string))

# Computing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clean_train_data = pickle.load(open('data/si_tr_s.pkl', 'rb'))
clean_valid_data = pickle.load(open('data/si_dt_05.pkl', 'rb'))

train_dataset = CleanSpeech(clean_train_data)
valid_dataset = CleanSpeech(clean_valid_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
    drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
    drop_last=True)

# Retraining an existing model 
if retrain:
    with open('models/stcn_{}.pkl'.format(time_string), 'rb') as f: 
        tcn_channels, tcn_kernel, tcn_res, concat_z, num_enc_layers, \
        latent_channels, dec_channels, dec_kernal, dropout, activation \
            = pickle.load(f)

    model = STCN(tcn_channels, tcn_kernel, tcn_res, concat_z, 
        num_enc_layers, latent_channels, dec_channels, dec_kernal, dropout, 
        activation).to(device)  

    model.load_state_dict(torch.load(stcn_path))

# Create new model
else:
    model = STCN(tcn_channels, tcn_kernel, tcn_res, concat_z, 
        num_enc_layers, latent_channels, dec_channels, dec_kernal, dropout, 
        activation).to(device)  
    start_epoch = 1

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define Hann window function for STFT 
hann = torch.hann_window(1024, device=device)

batch_loss_vec = []; train_loss_vec = []; valid_loss_vec = []


############################## TRAINING ########################################

def loss_function(S, recon_S, mu_p, logvar_p, mu_q, logvar_q, annealing): 
    # Reconstruction loss
    N, F, L = S.shape  
    recon = 1/L*torch.sum(S/(recon_S) - torch.log(S/(recon_S)) - 1)
    
    # Kullback-Leibler divergence
    N, D, L =  logvar_q[-1].shape
    num_latent_layers = len(mu_q)

    KL = 0
    for i in range(num_latent_layers):
        N, D, L = mu_q[i].shape
        KL += (1/(2*N*L))*(torch.sum(logvar_p[i]-logvar_q[i] 
                + logvar_q[i].exp()/logvar_p[i].exp() 
                + ((mu_p[i]-mu_q[i])**2)/(logvar_p[i].exp()))-D)
    
    return recon + annealing*KL, recon, KL

def train(epoch):
    model.train()
    train_loss = 0; batch_loss = 0; recon_loss = 0; KL_loss = 0
    annealing = min(1.0, float(epoch)/float(num_annealing)) 
    for batch_idx, data in enumerate(train_loader):

        # Clear gradients
        optimizer.zero_grad()

        s = data.to(device)
        S = torch.clamp(torch.abs(torch.stft(s, n_fft=1024, hop_length=256, 
            win_length=1024, window=hann, center=True, pad_mode='reflect',  
            normalized=False, onesided=None, return_complex=True))**2, 
            min=min_spec_energy)
        recon_S, mu_p, logvar_p, mu_q, logvar_q = model(S)
        
        # Regularization with standard normal prior
        if standard_normal_prior:
            mu_p = []; logvar_p = []
            for i in range(len(mu_q)):
                mu_p += [torch.zeros(mu_q[i].shape, device=device)]
                logvar_p += [torch.zeros(logvar_q[i].shape, device=device)]

        loss, recon, KL = loss_function(S, recon_S, mu_p, 
            logvar_p, mu_q, logvar_q, annealing)
        
        # Backward pass
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        batch_loss_vec.append(loss.item()/batch_size)

        if verbose and (batch_idx + 1) % batch_verbose == 0: 
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

            s = data.to(device)
            S = torch.clamp(torch.abs(torch.stft(s, n_fft=1024, hop_length=256, 
                win_length=1024, window=hann, center=True, pad_mode='reflect',  
                normalized=False, onesided=None, return_complex=True))**2, 
                min=min_spec_energy)
            recon_S, mu_p, logvar_p, mu_q, logvar_q = model(S)

            # Regularization with standard normal prior
            if standard_normal_prior:
                mu_p = []; logvar_p = []
                for i in range(len(mu_q)):
                    mu_p += [torch.zeros(mu_q[i].shape, device=device)]
                    logvar_p += [torch.zeros(logvar_q[i].shape, device=device)]
            else:
                for i in range(len(logvar_p)):
                    logvar_p[i] = torch.log(logvar_p[i].exp() + 0.1)

            loss, recon, KL = loss_function(S, recon_S, mu_p, logvar_p, mu_q, 
                logvar_q, 1.0)
            valid_loss += loss.item()
    return valid_loss

# Start training
for epoch in range(start_epoch, epochs + 1):

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
        with open('models/stcn_{}.pkl'.format(time_string), 'wb') as f:  
            pickle.dump([tcn_channels, tcn_kernel, tcn_res, concat_z,
                num_enc_layers, latent_channels, dec_channels, dec_kernal,
                dropout, activation], f)
