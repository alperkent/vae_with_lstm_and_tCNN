import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from model import VAE

# use GPU to train the model 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyperparameters
batch_size = 64
max_epochs = 25
learning_rate = 0.001
seq_length = 28
input_size = 28
hidden_size = 128
num_layers = 1

# datasets
train_set = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
val_set = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# loaders
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

# initialize the network, loss function and optimizer
network = VAE(seq_length, hidden_size, num_layers)
network.to(device)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

def criterion(outputs, inputs, mu, log_var):
    recon_loss = F.binary_cross_entropy(outputs, inputs.view(-1, 1, seq_length, input_size), reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
    loss = recon_loss + kld_loss
    return loss, recon_loss, kld_loss

# function to show an image
def imshow(img, title):
    npimg = np.transpose(img.cpu().numpy(), (1, 2, 0))
    plt.imshow(npimg)
    plt.title(title)
    plt.axis('off')
    plt.show()

# loop over epochs
train_loss = {'total': [], 'recon': [], 'kld': []}
val_loss = {'total': [], 'recon': [], 'kld': []}
for epoch in range(max_epochs):
    # training
    running_loss_train = {'total': 0.0, 'recon': 0.0, 'kld': 0.0}
    train_total = 0
    for inputs, labels in trainloader:
        inputs = inputs.view(-1, seq_length, input_size).to(device)
        optimizer.zero_grad()
        outputs, mu, log_var = network(inputs)
        loss, recon_loss, kld_loss = criterion(outputs, inputs, mu, log_var)
        loss.backward()
        optimizer.step()
        running_loss_train['total'] += loss.item()
        running_loss_train['recon'] += recon_loss.item()
        running_loss_train['kld'] += kld_loss.item()
        train_total += inputs.size(0)
        if train_total == 60000:
            imshow(torchvision.utils.make_grid(inputs.view(-1, 1, seq_length, input_size)), 'Inputs at epoch ' + str(epoch+1))
            imshow(torchvision.utils.make_grid(outputs), 'Outputs at epoch ' + str(epoch+1))
    train_loss['total'].append(running_loss_train['total']/train_total)
    train_loss['recon'].append(running_loss_train['recon']/train_total)
    train_loss['kld'].append(running_loss_train['kld']/train_total)
        
    # validation
    running_loss_val = {'total': 0.0, 'recon': 0.0, 'kld': 0.0}
    val_total = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs = inputs.view(-1, seq_length, input_size).to(device)
            outputs, mu, log_var = network(inputs)
            loss_val, recon_loss_val, kld_loss_val = criterion(outputs, inputs, mu, log_var)
            running_loss_val['total'] += loss_val.item()
            running_loss_val['recon'] += recon_loss_val.item()
            running_loss_val['kld'] += kld_loss_val.item()
            val_total += labels.size(0)
    val_loss['total'].append(running_loss_val['total']/val_total)
    val_loss['recon'].append(running_loss_val['recon']/val_total)
    val_loss['kld'].append(running_loss_val['kld']/val_total)

    print('Training loss at epoch %20d: %f' % (epoch+1, train_loss['total'][-1]))
    print('Training reconstruction loss at epoch %5d: %f' % (epoch+1, train_loss['recon'][-1]))
    print('Training KLD loss at epoch %16d: %f' % (epoch+1, train_loss['kld'][-1]))
    print('Validation loss at epoch %18d: %f' % (epoch+1, val_loss['total'][-1]))
    print('Validation reconstruction loss at epoch %3d: %f' % (epoch+1, val_loss['recon'][-1]))
    print('Validation KLD loss at epoch %14d: %f' % (epoch+1, val_loss['kld'][-1]))
    print('-------------------------------------------------------')

torch.save(network.state_dict(), 'model.pk')

# plot losses over epochs for training
plt.plot(range(1, max_epochs+1), train_loss['total'], label='Total loss')
plt.plot(range(1, max_epochs+1), train_loss['recon'], label='Recon. loss')
plt.plot(range(1, max_epochs+1), train_loss['kld'], label='KLD loss')
plt.title('Training losses per epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot losses over epochs for validation
plt.plot(range(1, max_epochs+1), val_loss['total'], label='Total loss')
plt.plot(range(1, max_epochs+1), val_loss['recon'], label='Recon. loss')
plt.plot(range(1, max_epochs+1), val_loss['kld'], label='KLD loss')
plt.title('Validation Losses per epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()