import torch
import pickle

from torch.utils.data import DataLoader
from dataset import SpectrogramFrames
from vae import VAE, loss_function
from utils import count_parameters


############################ SETTINGS ##########################################

batch_size = 128
learning_rate = 1e-3
epochs = 200
input_dim = 513
hid_dim = 128
latent_dim = 16
num_hid_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('System Information')
print('- Torch version: {}'.format(torch.__version__))
print('- Device: {}'.format(device))


########################## CONFIGURATION #######################################

print('Load data')
train_data = pickle.load(open('data/si_tr_s_frames.p', 'rb'))
valid_data = pickle.load(open('data/si_dt_05_frames.p', 'rb'))

train_dataset = SpectrogramFrames(train_data)
valid_dataset = SpectrogramFrames(valid_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

print('- Number of training samples: {}'.format(len(train_dataset)))
print('- Number of validation samples: {}'.format(len(valid_dataset)))

print('Create model')
model = VAE(in_out_dim=input_dim, hid_dim=hid_dim, latent_dim=latent_dim, 
    num_hid_layers=num_hid_layers).to(device)

print('- Number of learnable parameters: {}'.format(count_parameters(model)))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


############################## TRAINING ########################################

print('Start training')
def train():
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        x = data.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = loss_function(x, recon_x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss
    

def validate():
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(valid_loader):
            x = data.to(device)
            recon_x, mu, logvar = model(x)
            loss = loss_function(x, recon_x, mu, logvar)
            valid_loss += loss.item()
    return valid_loss


for epoch in range(1, epochs + 1):
    train_loss = train()
    valid_loss = validate()

    print('- Epoch: {:3d}    Train loss: {:4d}   Validation loss: {:4d}'.format(
          epoch, int(train_loss/len(train_loader.dataset)), 
          int(valid_loss/len(valid_loader.dataset))))

    torch.save(model.state_dict(), 'models/vae_{:03d}_vloss_{:04d}.pt'.format(
        epoch, int(valid_loss/len(valid_loader.dataset))))
