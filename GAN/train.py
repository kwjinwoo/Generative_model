import torch
import os
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.model import Discriminator, Generator
from tqdm import tqdm


BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
NOISE_DIM = 100
SAVE_DIR = "./saved_model/"
os.makedirs(SAVE_DIR, exist_ok=True)

train_dataset = datasets.MNIST(root="./data/",
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())
valid_dataset = datasets.MNIST(root="./data/",
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

generator = Generator(NOISE_DIM)
generator.to(device)
discriminator = Discriminator()
discriminator.to(device)

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    gen_mean_loss = 0
    dis_mean_loss = 0
    for x, _ in tqdm(train_loader):
        x = x.to(device)
        x = torch.where(x == 0., 0., 1.)

        noise = torch.randn([BATCH_SIZE, NOISE_DIM], device=device)

        generated = generator(noise)
        fake_output = discriminator(generated)
        real_output = discriminator(x)
        generator.zero_grad()
        gen_loss = generator.compute_loss(fake_output)
        generator_optimizer.zero_grad()
        gen_loss.backward(retain_graph=True)
        generator_optimizer.step()

        generated = generator(noise)
        fake_output = discriminator(generated)
        real_output = discriminator(x)
        discriminator.zero_grad()
        dis_loss = discriminator.compute_loss(real_output, fake_output)
        discriminator_optimizer.zero_grad()
        dis_loss.backward()
        discriminator_optimizer.step()

        gen_mean_loss += gen_loss.item()
        dis_mean_loss += dis_loss.item()

    print(f"EPOCH {epoch:>3d} gen_loss: {gen_mean_loss / len(train_loader):>6f} "
          f"dis_loss: {dis_mean_loss / len(train_loader)}")
    save_path = os.path.join(SAVE_DIR, "generator{}.pt".format(epoch))
    torch.save(generator.state_dict(), save_path)
