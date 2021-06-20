import torch
import torch.nn as nn

# use GPU to train the model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# create the model
class VAE(nn.Module):
    def __init__(self, seq_length, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # encoder
        self.lstm = nn.LSTM(seq_length, hidden_size, num_layers, batch_first=True)

        # sampler
        self.mean = nn.Linear(hidden_size, hidden_size//2)
        self.var = nn.Linear(hidden_size, hidden_size//2)

        # decoder with 4 layers
        # self.conv1 = nn.ConvTranspose2d(16, 8, 9)   # (2, 2) -> (10, 10)
        # self.bn1 = nn.BatchNorm2d(8)
        # self.conv2 = nn.ConvTranspose2d(8, 4, 9)    # (10, 10) -> (18, 18)
        # self.bn2 = nn.BatchNorm2d(4)
        # self.conv3 = nn.ConvTranspose2d(4, 2, 7)    # (18, 18) -> (24, 24)
        # self.bn3 = nn.BatchNorm2d(2)
        # self.conv4 = nn.ConvTranspose2d(2, 1, 5)    # (24, 24) -> (28, 28)
        
        # decoder with 5 layers
        # self.conv1 = nn.ConvTranspose2d(16, 12, 9)  # (2, 2) -> (10, 10)
        # self.bn1 = nn.BatchNorm2d(12)
        # self.conv2 = nn.ConvTranspose2d(12, 8, 7)   # (10, 10) -> (16, 16)
        # self.bn2 = nn.BatchNorm2d(8)
        # self.conv3 = nn.ConvTranspose2d(8, 4, 7)    # (16, 16) -> (22, 22)
        # self.bn3 = nn.BatchNorm2d(4)
        # self.conv4 = nn.ConvTranspose2d(4, 2, 5)    # (22, 22) -> (26, 26)
        # self.bn4 = nn.BatchNorm2d(2)
        # self.conv5 = nn.ConvTranspose2d(2, 1, 3)    # (26, 26) -> (28, 28)

        # decoder with 8 layers
        # self.conv1 = nn.ConvTranspose2d(16, 14, 5)  # (2, 2) -> (6, 6)
        # self.bn1 = nn.BatchNorm2d(14)
        # self.conv2 = nn.ConvTranspose2d(14, 12, 5)  # (6, 6) -> (10, 10)
        # self.bn2 = nn.BatchNorm2d(12)
        # self.conv3 = nn.ConvTranspose2d(12, 10, 5)  # (10, 10) -> (14, 14)
        # self.bn3 = nn.BatchNorm2d(10)
        # self.conv4 = nn.ConvTranspose2d(10, 8, 5)   # (14, 14) -> (18, 18)
        # self.bn4 = nn.BatchNorm2d(8)
        # self.conv5 = nn.ConvTranspose2d(8, 6, 5)    # (18, 18) -> (22, 22)
        # self.bn5 = nn.BatchNorm2d(6)
        # self.conv6 = nn.ConvTranspose2d(6, 4, 3)    # (22, 22) -> (24, 24)
        # self.bn6 = nn.BatchNorm2d(4)
        # self.conv7 = nn.ConvTranspose2d(4, 2, 3)    # (24, 24) -> (26, 26)
        # self.bn7 = nn.BatchNorm2d(2)
        # self.conv8 = nn.ConvTranspose2d(2, 1, 3)    # (26, 26) -> (28, 28)

        # decoder with 6 layers
        # self.conv1 = nn.ConvTranspose2d(16, 13, 7)  # (2, 2) -> (8, 8)
        # self.bn1 = nn.BatchNorm2d(13)
        # self.conv2 = nn.ConvTranspose2d(13, 10, 7)   # (8, 8) -> (14, 14)
        # self.bn2 = nn.BatchNorm2d(10)
        # self.conv3 = nn.ConvTranspose2d(10, 7, 5)    # (14, 14) -> (18, 18)
        # self.bn3 = nn.BatchNorm2d(7)
        # self.conv4 = nn.ConvTranspose2d(7, 4, 5)    # (18, 18) -> (22, 22)
        # self.bn4 = nn.BatchNorm2d(4)
        # self.conv5 = nn.ConvTranspose2d(4, 2, 5)    # (22, 22) -> (26, 26)
        # self.bn5 = nn.BatchNorm2d(2)
        # self.conv6 = nn.ConvTranspose2d(2, 1, 3)    # (26, 26) -> (28, 28)

        # decoder with 6 layers v2
        self.conv1 = nn.ConvTranspose2d(16, 13, 9)  # (2, 2) -> (10, 10)
        self.bn1 = nn.BatchNorm2d(13)
        self.conv2 = nn.ConvTranspose2d(13, 10, 7)   # (10, 10) -> (16, 16)
        self.bn2 = nn.BatchNorm2d(10)
        self.conv3 = nn.ConvTranspose2d(10, 7, 5)    # (16, 16) -> (20, 20)
        self.bn3 = nn.BatchNorm2d(7)
        self.conv4 = nn.ConvTranspose2d(7, 4, 5)    # (20, 20) -> (24, 24)
        self.bn4 = nn.BatchNorm2d(4)
        self.conv5 = nn.ConvTranspose2d(4, 2, 3)    # (24, 24) -> (26, 26)
        self.bn5 = nn.BatchNorm2d(2)
        self.conv6 = nn.ConvTranspose2d(2, 1, 3)    # (26, 26) -> (28, 28)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def sampler(self, mu, log_var):
        # apply reparametrization trick and samples from encoded distribution
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        z = mu + (eps * sigma)
        return z

    def forward(self, x):
        # set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # encode through LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]

        # sample with linear layer
        mu = self.mean(out)
        log_var = self.var(out)
        
        # sample without linear layer
        # out = out.reshape((-1, 2, self.hidden_size//2))
        # mu = out[:, 0, :]
        # log_var = out[:, 1, :]
        
        z = self.sampler(mu, log_var)

        # reshape for transpose convolution
        z = z.reshape((-1, self.hidden_size//8, 2, 2))

        # decode the sample with batchnorm
        # z = self.relu(self.bn1(self.conv1(z)))
        # z = self.relu(self.bn2(self.conv2(z)))
        # z = self.relu(self.bn3(self.conv3(z)))
        # z = self.relu(self.bn4(self.conv4(z)))
        # z = self.relu(self.bn5(self.conv5(z)))
        # z = self.sigmoid(self.conv6(z))
        
        # decode the sample without batchnorm
        z = self.relu(self.conv1(z))
        z = self.relu(self.conv2(z))
        z = self.relu(self.conv3(z))
        z = self.relu(self.conv4(z))
        z = self.relu(self.conv5(z))
        z = self.sigmoid(self.conv6(z))
        return z, mu, log_var
    
    def generate(self, z):
        # reshape for transpose convolution
        z = z.reshape((-1, self.hidden_size//8, 2, 2))

        # decode the sample with batchnorm
        # z = self.relu(self.bn1(self.conv1(z)))
        # z = self.relu(self.bn2(self.conv2(z)))
        # z = self.relu(self.bn3(self.conv3(z)))
        # z = self.relu(self.bn4(self.conv4(z)))
        # z = self.relu(self.bn5(self.conv5(z)))
        # z = self.sigmoid(self.conv6(z))
        
        # decode the sample without batchnorm
        z = self.relu(self.conv1(z))
        z = self.relu(self.conv2(z))
        z = self.relu(self.conv3(z))
        z = self.relu(self.conv4(z))
        z = self.relu(self.conv5(z))
        z = self.sigmoid(self.conv6(z))
        return z