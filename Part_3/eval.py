import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import pickle
import numpy as np

latent_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generatorfile = open('models/discriminator')
generator_network = pickle.load(generatorfile)

fixed_noise = torch.randn(64, latent_size, 1, 1, device=device)


gen_batch = generator_network(fixed_noise = torch.randn(64, latent_size, 1, 1, device=device))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(gen_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig('images/generated.png')