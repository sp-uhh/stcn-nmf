import torch
import pickle
import numpy as np

from vae import VAE
from stcn import STCN
from mcem import MCEM
from utils import count_parameters
from librosa import stft, istft



########################## CONFIGURATION #######################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_data = pickle.load(open('data/mixture.p', 'rb'))


############################## VAE #############################################

vae_model_path = 'models/vae_200_vloss_0476.pt'

vae = VAE(in_out_dim=513, hid_dim=128, latent_dim=16, 
    num_hid_layers=2).to(device)
vae.load_state_dict(torch.load(vae_model_path))
vae.eval()
for param in vae.parameters(): param.requires_grad = False

s_hat = []
win = np.sin(np.arange(.5,1024-.5+1)/1024*np.pi) 

for i, x in enumerate(test_data):
    print('File {}/{}'.format(i+1,len(test_data)), end="\r")
    x = x/np.max(x)
    T_orig = len(x)
    
    Y = stft(x, n_fft=1024, hop_length=256, win_length=1024, window=win)
    
    mcem = MCEM(Y, vae, device)
    mcem.run()
    mcem.separate(niter_MH=100, burnin=75)

    S_hat = mcem.S_hat + np.finfo(np.float32).eps
    s_hat.append(istft(stft_matrix=S_hat, hop_length=256, win_length=1024, 
        window=win, length=T_orig))
    
pickle.dump(s_hat, open('data/pickle/s_hat_vae.p', 'wb'), protocol=4)


############################## STCN ############################################

stcn_model_path = 'models/stcn_200_vloss_0_109.pt'

stcn = STCN(input_dim=513, tcn_channels=[64, 32, 16, 8], 
    latent_channels=[32, 16, 8, 4]).to(device)  

stcn.load_state_dict(torch.load(stcn_model_path))
stcn.eval()
for param in stcn.parameters(): param.requires_grad = False

s_hat = []
win = np.sin(np.arange(.5,1024-.5+1)/1024*np.pi) 

for i, x in enumerate(test_data):
    print('File {}/{}'.format(i+1,len(test_data)), end="\r")
    x = x/np.max(x)
    T_orig = len(x)
    
    Y = stft(x, n_fft=1024, hop_length=256, win_length=1024, window=win)
    
    mcem = MCEM(Y, stcn, device)
    mcem.run()
    mcem.separate(niter_MH=100, burnin=75)

    S_hat = mcem.S_hat + np.finfo(np.float32).eps
    s_hat.append(istft(stft_matrix=S_hat, hop_length=256, win_length=1024, 
        window=win, length=T_orig))
    
pickle.dump(s_hat, open('data/pickle/s_hat_stcn.p', 'wb'), protocol=4)