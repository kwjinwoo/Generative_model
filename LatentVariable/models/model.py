import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(True)
        )
        self.linear = nn.Linear(in_features=7*7*64, out_features=latent_dim + latent_dim)

    def forward(self, inputs):
        x = self.layers(inputs)
        latent_variable = x.view(-1, 7*7*64)
        latent_variable = self.linear(latent_variable)
        return latent_variable


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.linear = nn.Linear(in_features=latent_dim, out_features=7*7*32)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, output_padding=1,
                               stride=2, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, output_padding=1,
                               stride=2, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, padding=1,
                               stride=1, bias=False)
        )

    def forward(self, inputs):
        x = self.linear(inputs)
        x = x.view(-1, 32, 7, 7)
        out = self.layers(x)
        return out


def reparameterizing(mean, log_var):
    eps = torch.randn(mean.size(), device=mean.device)
    return eps * torch.exp(log_var * 0.5) + mean


def log_normal_pdf(sample, mean, log_var):
    log2pi = torch.log(torch.tensor([2. * np.pi], device=mean.device))
    return torch.sum(
        -.5 * ((sample - mean) ** 2. * torch.exp(-log_var) + log_var + log2pi), 1)


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.latent_dim)
        self.sigmoid = nn.Sigmoid()

    def encoding(self, inputs):
        out = self.encoder(inputs)
        mean, log_var = torch.split(out, 2, 1)
        return mean, log_var

    def decoding(self, z):
        logit = self.decoder(z)
        return logit

    def sampling(self, num_samples):
        eps = torch.randn((num_samples, self.latent_dim)).cuda()
        return self.sigmoid(self.decoder(eps))

    def compute_loss(self, inputs):
        mean, log_var = self.encoding(inputs)
        z = reparameterizing(mean, log_var)
        x_logit = self.decoding(z)
        cross_ent = F.binary_cross_entropy_with_logits(x_logit, inputs, reduction="none")
        logpx_z = -torch.sum(cross_ent, (1, 2, 3))
        logpz = log_normal_pdf(z, torch.tensor([0.], device=mean.device),
                               torch.tensor([0.], device=mean.device))
        logqz_x = log_normal_pdf(z, mean, log_var)
        return -torch.mean(logpx_z + logpz - logqz_x), self.sigmoid(x_logit)

    def forward(self, inputs):
        _, out = self.compute_loss(inputs)
        return out


if __name__ == "__main__":
    vae = VAE(2)
    print(vae)
