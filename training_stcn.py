import torch
import pickle

from torch.utils.data import DataLoader
from dataset import Spectrogram, collate_fn
from stcn import STCN, loss_function
from utils import count_parameters


############################ SETTINGS ##########################################

batch_size = 16
learning_rate = 1e-3
num_annealing = 20
epochs = 200
input_dim = 513
tcn_channels = [64, 32, 16, 8]
latent_channels = [32, 16, 8, 4]


########################## CONFIGURATION #######################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('System Information')
print('- Torch version: {}'.format(torch.__version__))
print('- Device: {}'.format(device))

print('Load data')
train_data = pickle.load(open('data/si_tr_s.p', 'rb'))
valid_data = pickle.load(open('data/si_dt_05.p', 'rb'))
    
train_dataset = Spectrogram(train_data)
valid_dataset = Spectrogram(valid_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
    collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
    collate_fn=collate_fn)

print('- Number of training samples: {}'.format(len(train_dataset)))
print('- Number of validation samples: {}'.format(len(valid_dataset)))

print('Create model')
model = STCN(input_dim, tcn_channels, latent_channels).to(device)  

print('- Number of learnable parameters: {}'.format(count_parameters(model)))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


############################## TRAINING ########################################

def train(epoch):
    model.train()
    train_loss = 0
    annealing = min(1.0, float(epoch)/float(num_annealing)) 
    for batch_idx, data in enumerate(train_loader):
        x = data.to(device)
        optimizer.zero_grad()
        recon_x, z_p, z_q, mu_p, logvar_p, mu_q, logvar_q = model(x)
        loss = loss_function(x, recon_x, mu_p, logvar_p, mu_q, logvar_q, 
            annealing)
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
            recon_x, z_p, z_q, mu_p, logvar_p, mu_q, logvar_q = model(x)
            loss = loss_function(x, recon_x, mu_p, logvar_p, mu_q, logvar_q, 
                1.0)
            valid_loss += loss.item()
    return valid_loss

print('Start training')
for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
    valid_loss = validate()

    print('- Epoch: {}    Train loss: {:.3f}   Validation loss: {:.3f}'.format(
        epoch, train_loss/len(train_loader.dataset), 
        valid_loss/len(valid_loader.dataset)))

    torch.save(model.state_dict(), 'models/stcn_{:03d}_vloss_{:.3f}.pt'.format(
        epoch, valid_loss / len(valid_loader.dataset)))
