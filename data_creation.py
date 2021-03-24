import os
import pickle
import numpy as np

from glob import glob
from librosa import load
from librosa.core import resample
from librosa.output import write_wav


############################ SETTINGS ##########################################

fs = 16000
snrs = [-5.0, 0.0, 5.0]
split = 2 # slit audio files every 2 seconds

create_training_data = True
create_validation_data = True
create_test_data = True

########################## TRAINING DATA #######################################

if create_training_data:

    data_dir = '../vae-speech-modeling/data/CSR-1-WSJ-0/WAV/wsj0/si_tr_s'
    audio_paths = sorted(glob(data_dir + '/**/*.wav', recursive=True))

    audio_files = []

    # Load wav files and normalize each file
    for path in audio_paths:
        s, fs_s = load(path, sr=None) 
        s = s/np.max(np.abs(s))
        if fs_s != fs: raise ValueError('Unexpected sampling rate')
        audio_files.append(s)

    # Concatenate along time
    audio_files = np.concatenate(audio_files)

    # Split in audio files of same legth
    num_spilts = int(len(audio_files)/(fs*split))
    audio_files = np.split(audio_files[:fs*split*num_spilts], num_spilts)

    # Save as pickle file
    pickle.dump(audio_files, open('data/si_tr_s.pkl', 'wb'), protocol=4)


########################## VALIDATION DATA #####################################

if create_validation_data:

    data_dir = '../vae-speech-modeling/data/CSR-1-WSJ-0/WAV/wsj0/si_dt_05'
    audio_paths = sorted(glob(data_dir + '/**/*.wav', recursive=True))

    audio_files = []

    # Load wav files and normalize each file
    for path in audio_paths:
        s, fs_s = load(path, sr=None) 
        s = s/np.max(np.abs(s))
        if fs_s != fs: raise ValueError('Unexpected sampling rate')
        audio_files.append(s)

    # Concatenate along time
    audio_files = np.concatenate(audio_files)

    # Split in audio files of same legth
    num_spilts = int(len(audio_files)/(fs*split))
    audio_files = np.split(audio_files[:fs*split*num_spilts], num_spilts)

    # Save as pickle file
    pickle.dump(audio_files, open('data/si_dt_05.pkl', 'wb'), protocol=4)


########################## TEST DATA ###########################################

if create_test_data:

    # Load noise
    data_dir = '../vae-speech-modeling/data/QUT-NOISE/QUT-NOISE/'

    types = ['cafe', 'home', 'street', 'car']

    cafe, fs_cafe = load(os.path.join(data_dir, 'CAFE-CAFE-1.wav'), sr=None) 
    car, fs_car = load(os.path.join(data_dir, 'CAR-WINDOWNB-1.wav'), sr=None)
    home, fs_home = load(os.path.join(data_dir, 'HOME-KITCHEN-1.wav'), sr=None) 
    street, fs_street = load(os.path.join(data_dir, 'STREET-CITY-1.wav'), sr=None)

    # Resample noise data to match speech data
    cafe = resample(cafe, fs_cafe, fs)
    car = resample(car, fs_car, fs)
    home = resample(home, fs_home, fs)
    street = resample(street, fs_street, fs)

    # Load speech data
    data_dir = '../vae-speech-modeling/data/CSR-1-WSJ-0/WAV/wsj0/si_et_05'
    audio_paths = sorted(glob(data_dir + '/**/*.wav', recursive=True))

    audio_files = []

    # Load wav files and normalize each file
    for i, path in enumerate(audio_paths):
        s, fs_s = load(path, sr=None) 
        s = s/np.max(np.abs(s))
        if fs_s != fs: raise ValueError('Unexpected sampling rate')  
        audio_files.append(s)
            
    pickle.dump(audio_files, open('data/clean_speech.pkl', 'wb'), protocol=4)

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
        noises.append((noise))
        mixture = speech + noise
        mixtures.append(mixture)
        
    pickle.dump(noises, open('data/noise.pkl', 'wb'), protocol=4)
    pickle.dump(mixtures, open('data/mixture.pkl', 'wb'), protocol=4)