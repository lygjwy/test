import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaAE(nn.Module):
    def __init__(self):
        super(VanillaAE, self).__init__()

        # 3 * 32 * 32 = 3072
        # 1 * 28 * 28 = 784
        self.fc1 = nn.Linear(3072, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 2048)
        self.fc6 = nn.Linear(2048, 3072)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = self.fc3(x)
        return z
        # return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        x = F.relu(self.fc4(z))
        x = F.relu(self.fc5(x))
        x = F.sigmoid(self.fc6(x))
        return x

    def forward(self, x):
        # mu, logvar = self.encode(x.view(-1, 3072))
        # z = self.reparameterize(mu, logvar)
        z = self.encode(x.view(-1, 3072))
        return self.decode(z)
    

def get_vanilla_ae():
    return VanillaAE()