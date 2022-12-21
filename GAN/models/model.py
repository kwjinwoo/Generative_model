import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()

        self.input_dim = input_dim
        self.input_layer = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=7*7*256, bias=False),
            nn.BatchNorm1d(7*7*256),
            nn.ReLU(True)
        )
        self.conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=5, stride=2, padding=2,
                               output_padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.input_layer(inputs)
        x = x.view(-1, 256, 7, 7)
        out = self.conv_layer(x)
        return out

    @staticmethod
    def compute_loss(fake_output):
        total_loss = F.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output))
        return total_loss


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(in_features=7 * 7 * 128, out_features=1)

    def forward(self, inputs):
        x = self.conv_layers(inputs)
        x = x.view(-1, 7 * 7 * 128)
        out = self.classifier(x)
        return out

    @staticmethod
    def compute_loss(real_output, fake_output):
        real_loss = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output))
        fake_loss = F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))
        total_loss = real_loss + fake_loss
        return total_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = Generator(100)
    d = Discriminator()
    from torchinfo import summary

    g = g.to(device)
    d = d.to(device)
    print(summary(g, (4, 100)))
    print(summary(d, (1, 1, 28, 28)))
