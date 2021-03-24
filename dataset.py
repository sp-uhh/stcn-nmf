import torch
import numpy as np

from librosa import stft
from torch.utils.data import Dataset


# STFT parameters

fs = int(16e3) 
wlen_sec = 64e-3 
wlen = np.int(np.power(2, np.ceil(np.log2(wlen_sec*fs))))
win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi)
hop_percent = 0.25 
hop = np.int(hop_percent*wlen) 
nfft = wlen


class CleanSpeech(Dataset):
    def __init__(self, data):
        self.data = data
        self.index = np.arange(len(self.data))
 
    def __getitem__(self, i):
        return self.data[i]
    
    def __len__(self):
        return len(self.data)

class Spectrogram(Dataset):
    def __init__(self, data):
        self.data = data
        self.index = np.arange(len(self.data))
 
    def __getitem__(self, i):
        return np.power(np.abs(stft(self.data[i], n_fft=nfft, hop_length=hop, 
                                            win_length=wlen, window=win)), 2)
    
    def __len__(self):
        return len(self.data)

def collate_fn(data):    
    max_length = max(sample.shape[1] for sample in data)      
    batch = []
    for sample in data:
        batch.append(np.pad(sample, ((0, 0), (0, max_length-sample.shape[1])), 'constant', constant_values=(1e-8, 1e-8)))
    return torch.tensor(batch)


class SpectrogramFrames(Dataset):
    def __init__(self, data):
        self.data = data
        self.index = np.arange(len(self.data))

    def __getitem__(self, i):
        return self.data[:,i]

    def __len__(self):
        return len(self.data[0,:])