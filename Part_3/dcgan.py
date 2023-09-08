#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

dataroot = "data"

workers = 2
batch_size = 128
image_size = 64
channels = 3
latent_size = 100
epochs = 5
learning_rate = 0.0002

ngpu=1

# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5


dataset = datasets.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Generator class
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=8*ngf, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(8*ngf),
            nn.ReLU(True)
            ,
            nn.ConvTranspose2d(8*ngf, 4*ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(4*ngf, 2*ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d( 2*ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh() # Output value between [-1, 1] to match normalized data
        )

    def forward(self, input):
        return self.main(input)
    

# Discriminator class
"""
Use strided convolution instead of pooling to downsample as the network learns its own pooling funciton. 
LeakyRelU instead of RelU apparently performs better.
"""
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, 2*ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2*ndf, 4*ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(4*ndf, 8*ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8*ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(8*ndf, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



# Create the generator
generator_network = Generator(ngpu).to(device)
# Create the Discriminator
discriminator_network = Discriminator(ngpu).to(device)

real_image = 1
generated_image = 0

# Setup Adam optimizers for both Generator and Discriminator
optimizerD = optim.Adam(discriminator_network.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator_network.parameters(), lr=learning_rate, betas=(beta1, 0.999))

