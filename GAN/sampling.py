import matplotlib.pyplot as plt
import torch
import os
from models.model import Generator


NOISE_DIM = 100
MODEL_PATH = "./saved_model/generator49.pt"
SAVE_DIR = "./assets/"
os.makedirs(SAVE_DIR, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Generator(NOISE_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

seed = torch.randn([16, NOISE_DIM], device=device)\

with torch.no_grad():
    generated = model(seed)
generated = torch.where(generated > 0.5, 1., 0.)

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated[i].permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")
plt.suptitle("GAN generated samples")
plt.savefig(os.path.join(SAVE_DIR, "GAN_generate.png"))
