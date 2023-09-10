import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import pickle
import numpy as np

print("> Starting generation script")
latent_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generatorfile = open('/home/Student/s4501559/Dev/COMP3710-Demo-2/Part_3/models/generator')
generator_network = pickle.load(generatorfile)
print("> Model loaded")

fixed_noise = torch.randn(64, latent_size, 1, 1, device=device)

print("> Generating batch")
gen_batch = generator_network(fixed_noise = torch.randn(64, latent_size, 1, 1, device=device))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(gen_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig('/home/Student/s4501559/Dev/COMP3710-Demo-2/Part_3/images/generated.png')
print("> Images saved to images/generated.png")
