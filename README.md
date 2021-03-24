#  VAE and STCN implementations for single-channel speech enhancement 

This repository contains the non-sequential VAE and STCN speech models and the NMF noise model for single-channel speech enhancement. 

3/18/2021: VAE is now instance of STCN when the parameters are set to: 
```python 
kernel_size = 1
tcn_channels = [128]
latent_channels = [16]
dec_channels = [16, 128, 128, 513]
concat_z = False
```
3/24/2021: STFT is being calculated on GPU 

Whenever you use this code for any experiments and/or publications you need to cite our original paper [1].

[1] Julius Richter, Guillaume Carbajal, Timo Gerkmann, "Speech Enhancement with Stochastic Temporal Convolutional Networks", Proc. Interspeech 2020, DOI: 10.21437/Interspeech.2020-2588.