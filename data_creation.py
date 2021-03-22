import os
import sys
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt 

from librosa import load, stft
from librosa.core import resample
from librosa.output import write_wav


############################ SETTINGS ##########################################

fs = int(16e3)
wlen_sec = 64e-3 
hop_percent = 0.25 
wlen = wlen_sec*fs 
wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) 
hop = np.int(hop_percent*wlen) 
nfft = wlen
win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi)
snrs = [-5.0, 0.0, 5.0]
split = 2 # slit audio files every 2 second

########################## TRAINING DATA #######################################

data_dir = 'data/CSR-1-WSJ-0/WAV/wsj0/si_tr_s'
folders = sorted(os.listdir(data_dir))
audio_files = []
spectrograms = []

for folder in folders:
    files = sorted(os.listdir(os.path.join(data_dir, folder)))
    for file in files:
        path = os.path.join(os.path.join(data_dir, folder), file)
        x, fs_x = load(path, sr=None) 
        x = x/np.max(np.abs(x))
        if fs != fs_x: raise ValueError('Unexpected sampling rate')
        audio_files.append(x)
        spectrograms.append(np.power(np.abs(stft(x, n_fft=nfft, hop_length=hop, 
                                            win_length=wlen, window=win)), 2))

sorted_audio_files = sorted(audio_files, key=len) 

audio_files_cat = np.concatenate(audio_files)
num_spilts = int(len(audio_files_cat)/(fs*split))
audio_files_split = np.array_split(audio_files_cat, num_spilts)

pickle.dump(audio_files_split, open('data/si_tr_s_split.p', 'wb'), protocol=4)

spectrograms = np.concatenate(spectrograms, axis=1)      
pickle.dump(spectrograms, open('data/si_tr_s_frames.p', 'wb'), protocol=4)


########################## VALIDATION DATA #####################################

data_dir = 'data/CSR-1-WSJ-0/WAV/wsj0/si_dt_05'
folders = sorted(os.listdir(data_dir))
audio_files = []
spectrograms = []

for folder in folders:
    files = sorted(os.listdir(os.path.join(data_dir, folder)))
    for file in files:
        path = os.path.join(os.path.join(data_dir, folder), file)
        x, fs_x = load(path, sr=None)
        x = x/np.max(np.abs(x))
        if fs != fs_x: raise ValueError('Unexpected sampling rate')
        audio_files.append(x)
        spectrograms.append(np.power(np.abs(stft(x, n_fft=nfft, hop_length=hop, 
                                            win_length=wlen, window=win)), 2))


audio_files_cat = np.concatenate(audio_files)
num_spilts = int(len(audio_files_cat)/(fs*split))
audio_files_split = np.array_split(audio_files_cat, num_spilts)

pickle.dump(audio_files_split, open('data/si_dt_05_split.p', 'wb'), protocol=4)

spectrograms = np.concatenate(spectrograms, axis=1)
pickle.dump(spectrograms, open('data/si_dt_05_frames.p', 'wb'), protocol=4)


########################## TEST DATA ###########################################

# Load noise
data_dir = 'data/QUT-NOISE/QUT-NOISE/'

cafe, fs_cafe = load(os.path.join(data_dir, 'CAFE-CAFE-1.wav'), sr=None) 
car, fs_car = load(os.path.join(data_dir, 'CAR-WINDOWNB-1.wav'), sr=None)
home, fs_home = load(os.path.join(data_dir, 'HOME-KITCHEN-1.wav'), sr=None) 
street, fs_street = load(os.path.join(data_dir, 'STREET-CITY-1.wav'), sr=None)

cafe = resample(cafe, fs_cafe, fs)
car = resample(car, fs_car, fs)
home = resample(home, fs_home, fs)
street = resample(street, fs_street, fs)

# Load speech
data_dir = 'data/CSR-1-WSJ-0/WAV/wsj0/si_et_05/'
folders = sorted(os.listdir(data_dir))
audio_files = []

for folder in folders:
    files = sorted(os.listdir(os.path.join(data_dir, folder)))
    for file in files:
        path = os.path.join(os.path.join(data_dir, folder), file)
        x, fs_x = load(path, sr=None) 
        x = x/np.max(np.abs(x))
        if fs != fs_x: raise ValueError('Unexpected sampling rate')
        audio_files.append(x)
        
pickle.dump(audio_files, open('data/clean_speech.p', 'wb'), protocol=4)

# Create mixtures
np.random.seed(0)
noise_types = np.random.randint(4, size=len(audio_files))
snrs_index = np.random.randint(3, size=len(audio_files))
mixtures = []
noises = []

for i, speech in enumerate(audio_files):
    noise_type = noise_types[i]
    snr_dB = snrs[snrs_index[i]]
    speech_power = 1/len(speech)*np.sum(np.power(speech, 2))
    
    if noise_type == 0:
        start = np.random.randint(len(cafe)-len(speech))
        noise = cafe[start:start+len(speech)]
    elif noise_type == 1:
        start = np.random.randint(len(home)-len(speech))
        noise = home[start:start+len(speech)]
    elif noise_type == 2:
        start = np.random.randint(len(street)-len(speech))
        noise = street[start:start+len(speech)]
    elif noise_type == 3:
        start = np.random.randint(len(car)-len(speech))
        noise = car[start:start+len(speech)]
    else:
        raise ValueError('Unexpected noise type index')
    noises.append(noise)
    noise_power = 1/len(speech)*np.sum(np.power(noise, 2))
    noise_power_target = speech_power*np.power(10,-snr_dB/10)
    k = noise_power_target / noise_power
    noise = noise * np.sqrt(k)
    mixtures.append((speech+noise))
    
pickle.dump(mixtures, open('data/mixture.p', 'wb'), protocol=4)
