import sys
import torch
import pickle
import numpy as np

from stcn import STCN
from mcem import MCEM
from glob import glob
from librosa import stft, istft


model_path = 'models/stcn_2021-03-24_21:36:01_001_vloss_758.pt'
eval_stcn = True

########################## CONFIGURATION #######################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_data = pickle.load(open('data/mixture.pkl', 'rb'))

time_stemp = model_path[0:31]

with open('{}.pkl'.format(time_stemp), 'rb') as f: 
    tcn_channels, tcn_kernel, tcn_res, concat_z, num_enc_layers, \
        latent_channels, dec_channels, dec_kernal, dropout, activation = pickle.load(f)

stcn = STCN(tcn_channels, tcn_kernel, tcn_res, concat_z, 
        num_enc_layers, latent_channels, dec_channels, dec_kernal, dropout, 
        activation).to(device)  

stcn.load_state_dict(torch.load(model_path))
stcn.eval()
for param in stcn.parameters(): param.requires_grad = False

s_hat_stcn = []

############################ ENHANCEMENT #######################################

for i, x in enumerate(test_data):
    print('Enhance File {}/{}'.format(i+1,len(test_data)), end="\r")
    T_orig = len(x)
    
    X = stft(x, 1024, 256, 1024, np.hanning(1024))

    mcem = MCEM(X, stcn, device)
    mcem.run()
    mcem.separate(niter_MH=100, burnin=75)
    S_hat_stcn = mcem.S_hat + np.finfo(np.float32).eps
    s_hat_stcn.append(istft(S_hat_stcn, 256, 1024, np.hanning(1024), length=T_orig))

pickle.dump(s_hat_stcn, open('data/s_hat_stcn.p', 'wb'), protocol=4)

