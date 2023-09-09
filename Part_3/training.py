#%matplotlib inline
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pickle

print("> Start training script")
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

real_label = 1
generated_label = 0

# Setup Adam optimizers for both Generator and Discriminator
optimizerD = optim.Adam(discriminator_network.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator_network.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Loss function
bce_loss = nn.BCELoss()

# Create batch of latent vectors
fixed_noise = torch.randn(64, latent_size, 1, 1, device=device)


# Training Loop
img_list = []
G_losses = []
D_losses = []
iters = 0

print("> Start training loop")

for epoch in range(epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        """ Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z))) """
        discriminator_network.zero_grad()
        X = data[0].to(device)

        b_size = X.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        # Forward pass real batch through Discriminator
        output = discriminator_network(X).view(-1)

        # Calculate loss on all-real batch
        discriminator_loss = bce_loss(output, label)
        discriminator_loss.backward()

        D_x = output.mean().item()

        ## Train with all-fake batch
        latent_vecs = torch.randn(b_size, latent_size, 1, 1, device=device)
        gen_X = generator_network(latent_vecs)

        label = torch.full((b_size,), generated_label, dtype=torch.float, device=device)

        output = discriminator_network(gen_X.detach()).view(-1)

        gen_loss = bce_loss(output, label)
        gen_loss.backward()

        D_G_z1 = output.mean().item()

        # Compute error of D as sum over the fake and the real batches
        errD = discriminator_loss + gen_loss

        # Update Discriminator
        optimizerD.step()

        # (2) Update G network: maximize log(D(G(z)))
   
        generator_network.zero_grad()
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device) # fake labels are real for generator cost
        output = discriminator_network(gen_X.detach()).view(-1)

        # Calculate G's loss based on this output
        gen_loss = bce_loss(output, label)

        # Calculate gradients for G
        gen_loss.backward()

        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     discriminator_loss.item(), gen_loss.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(gen_loss.item())
        D_losses.append(discriminator_loss.item())

print("> Saving models")
with open("models/discriminator", "w") as savefile:
    pickle.dump(discriminator_network, savefile)
    savefile.close()
    print("> Discriminator pickled to models/discriminator")

with open("models/generator", "w") as savefile:
    pickle.dump(generator_network, savefile)
    savefile.close()
    print("> Generator pickled to models/generator")



plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("images/lossfuncs.png")

print("END")