import matplotlib.pyplot as plt
import torch
import os
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.model import PixelCNN
from tqdm import tqdm

MODEL_PATH = "./saved_model/pixelcnn49.pt"
SAVE_DIR = "./assets/"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PixelCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)


generated = np.zeros((16, 1, 28, 28), dtype=np.float32)
generated = torch.from_numpy(generated)
generated = generated.to(device)
with torch.no_grad():
    for h in tqdm(range(28)):
        for w in range(28):
            out = model(generated)
            generated_pixel = torch.bernoulli(out[:, :, h, w])
            generated[:, :, h, w] = generated_pixel

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")
plt.suptitle("PixelCNN generated samples")
plt.savefig(os.path.join(SAVE_DIR, "pixelCNN_generate.png"))


valid_dataset = datasets.MNIST(root="../data/",
                               train=False,
                               download=True,
                               transform=transforms.ToTensor())
valid_loader = DataLoader(valid_dataset, batch_size=16)
generated = next(iter(valid_loader))[0]
generated = torch.where(generated == 0., 0., 1.).to(device)
generated[:, :, 14:, :] = 0
with torch.no_grad():
    for h in tqdm(range(14, 28)):
        for w in range(28):
            out = model(generated)
            generated_pixel = torch.bernoulli(out[:, :, h, w])
            generated[:, :, h, w] = generated_pixel * 2
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")
plt.suptitle("PixelCNN half generated samples")
plt.savefig(os.path.join(SAVE_DIR, "pixelCNN_half_generate.png"))
