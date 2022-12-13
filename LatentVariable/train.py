import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.model import VAE
from tqdm import tqdm


BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
LATENT_DIM = 2
SAVE_DIR = "./saved_model/"
os.makedirs(SAVE_DIR, exist_ok=True)


train_dataset = datasets.MNIST(root="../data/",
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())
valid_dataset = datasets.MNIST(root="../data/",
                               train=False,
                               download=True,
                               transform=transforms.ToTensor())

print("number of train data :", len(train_dataset))
print("number of valid data :", len(valid_dataset))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is :", device)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                          shuffle=False)

model = VAE(latent_dim=LATENT_DIM)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
for epoch in range(NUM_EPOCHS):
    mean_loss = 0
    for x, _ in tqdm(train_loader):
        x = x.to(device)
        x = torch.where(x == 0., 0., 1.)
        optimizer.zero_grad()
        loss, _ = model.compute_loss(x)
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()
    print(f"EPOCH {epoch:>3d} ELBO: {- mean_loss / len(train_loader):>6f}")

    save_path = os.path.join(SAVE_DIR, "VAE{}.pt".format(epoch))
    torch.save(model.state_dict(), save_path)
