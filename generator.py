import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from model import VAE

# use GPU to train the model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# function to show an image
def imshow(img, title):
    npimg = np.transpose(img.cpu().numpy(), (1, 2, 0))
    plt.imshow(npimg)
    plt.title(title)
    plt.axis('off')
    plt.show()

seq_length = 28
hidden_size = 128
num_layers = 1
network = VAE(seq_length, hidden_size, num_layers)
network.to(device)
network.load_state_dict(torch.load('model.pk'))

with torch.no_grad():
    vectors = torch.randn(100, hidden_size//2).to(device)
    images = network.generate(vectors)
    imshow(torchvision.utils.make_grid(images, normalize=False, nrow=10, padding=5), 'Generated Images')
