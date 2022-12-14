import torch
import os
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.model import PixelCNN
from tqdm import tqdm


BATCH_SIZE = 128
LEARNING_RATE = 0.0005
NUM_EPOCHS = 50
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

model = PixelCNN()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
bce_loss = nn.BCELoss()
for epoch in range(NUM_EPOCHS):
    mean_loss = 0
    for x, _ in tqdm(train_loader):
        x = x.to(device)
        x = torch.where(x == 0., 0., 1.)

        optimizer.zero_grad()
        out = model(x)
        loss = bce_loss(out, x)
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()
    print(f"EPOCH {epoch:>3d} loss: {mean_loss / len(train_loader):>6f}")

    save_path = os.path.join(SAVE_DIR, "pixelcnn{}.pt".format(epoch))
    torch.save(model.state_dict(), save_path)
