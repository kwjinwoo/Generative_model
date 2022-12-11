import torch
import torch.nn as nn


class MaskedConvolution(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MaskedConvolution, self).__init__(*args, **kwargs)
        self.mask = torch.zeros_like(self.weight.data,
                                     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # all valid condition
        yc, xc = self.weight.data.size()[-2] // 2, self.weight.data.size()[-1] // 2
        print(yc, xc)
        self.mask[..., :yc, :] = 1.
        self.mask[..., yc, :xc+1] = 1.
        print(self.mask.size())
        print(self.weight.data.size())

    def forward(self, x):
        self.weight.data *= self.mask
        out = super(MaskedConvolution, self).forward(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.conv1 = MaskedConvolution(in_channels=in_channel, out_channels=in_channel // 2,
                                       kernel_size=1, padding="same", stride=1, bias=False)
        self.conv2 = MaskedConvolution(in_channels=in_channel // 2, out_channels=in_channel // 2, kernel_size=3,
                                       padding="same", stride=1, bias=False)
        self.conv3 = MaskedConvolution(in_channels=in_channel // 2, out_channels=in_channel,
                                       kernel_size=1, padding="same", stride=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        out = x + residual
        return out


class PixelCNN(nn.Module):
    def __init__(self, h):
        super().__init__()

        self.layers = nn.Sequential(
            MaskedConvolution(in_channels=1, out_channels=h, kernel_size=7, padding="same", bias=False),
            nn.ReLU(True),
            MaskedConvolution(in_channels=h, out_channels=h, kernel_size=7, padding="same", bias=False),
            nn.ReLU(True),
            MaskedConvolution(in_channels=h, out_channels=h, kernel_size=7, padding="same", bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=h, out_channels=2, kernel_size=1)
        )

    def forward(self, x):
        out = self.layers(x)
        return out

    def generate(self, num_samples, sample_size):
        w, h = sample_size
        generated_sample = torch.zeros((num_samples, 1, h, w))
        for i in range(w):
            for c in range(h):
                generated_sample = generated_sample.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                generated = self.forward(generated_sample).softmax(1)
                sample = torch.multinomial(generated[:, 0, c, i], 1)
                generated_sample[:, 0, c, i] = sample
        return generated_sample
# class PixelCNN(nn.Module):
#     def __init__(self, num_layers, h):
#         super().__init__()
#
#         self.conv1 = MaskedConvolution(in_channels=1, out_channels=h * 2, kernel_size=7,
#                                        padding="same", stride=1, bias=False)
#         modules = [ResidualBlock(in_channel=h * 2) for _ in range(num_layers)]
#         self.residual_modules = nn.Sequential(*modules)
#         self.out = MaskedConvolution(in_channels=h * 2, out_channels=1,
#                                      kernel_size=1, stride=1, padding="same", bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.residual_modules(x)
#         x = self.out(x)
#
#         out = self.sigmoid(x)
#         return out
#
#     def generate(self, num_samples, sample_size):
#         w, h = sample_size
#         generated_sample = torch.zeros((num_samples, 1, h, w))
#         for i in range(w):
#             for c in range(h):
#                 generated_sample = generated_sample.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#                 generated = self.forward(generated_sample).softmax(1)
#                 sample = torch.multinomial(generated[:, 0, c, i], 1)
#                 generated_sample[:, 0, c, i] = sample
#         return generated_sample
