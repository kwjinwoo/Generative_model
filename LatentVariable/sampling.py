import matplotlib.pyplot as plt
import torch
import os
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.model import VAE
from tqdm import tqdm


MODEL_PATH = "./saved_model/VAE9.pt"
SAVE_DIR = "./assets/"
LATENT_DIM = 2
os.makedirs(SAVE_DIR, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE(LATENT_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

eps = torch.randn((16, LATENT_DIM), device=device)
generated = model.sampling(eps)


for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")
plt.suptitle("VAE generated samples")
plt.savefig(os.path.join(SAVE_DIR, "VAE_generate.png"))
