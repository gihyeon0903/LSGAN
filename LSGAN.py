import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn3   = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.bn4   = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        return x.view(-1, 1)
    
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        
        self.tconv1 = nn.ConvTranspose2d(in_channels=z_dim, out_channels=1024, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1    = nn.BatchNorm1d(1024)
        self.tconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        self.bn2    = nn.BatchNorm1d(512)
        self.tconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.bn3    = nn.BatchNorm1d(256)
        self.tconv4 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn4    = nn.BatchNorm1d(128)
        self.tconv5 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))
        
        x = torch.tanh(self.tconv5(x))
        
        return x
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)