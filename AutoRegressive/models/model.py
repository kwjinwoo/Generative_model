import torch
import torch.nn as nn


class MaskedConvolution(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConvolution, self).__init__(*args, **kwargs)
        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        # all valid condition
        yc, xc = self.weight.data.size()[-2] // 2, self.weight.data.size()[-1] // 2
        self.mask[..., :yc, :] = 1.
        self.mask[..., yc, :xc+1] = 1.

        if mask_type == 'A':
            self.mask[..., yc, xc] = 0.

    def forward(self, x):
        self.weight.data *= self.mask
        out = super(MaskedConvolution, self).forward(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 2, kernel_size=1, padding="same", bias=False),
            nn.ReLU(inplace=True),
            MaskedConvolution(mask_type='B', in_channels=in_channel // 2, out_channels=in_channel // 2, kernel_size=3,
                              padding="same", bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channel // 2, out_channels=in_channel, kernel_size=1, padding="same", bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        x = self.layers(inputs)
        out = x + inputs
        return out


class PixelCNN(nn.Module):
    def __init__(self, num_filters=128, num_layers=5):
        super(PixelCNN, self).__init__()

        self.input_conv = nn.Sequential(
            MaskedConvolution(mask_type='A', in_channels=1, out_channels=num_filters, kernel_size=7,
                              padding="same", bias=False),
            nn.ReLU(True)
        )

        layers = [ResidualBlock(in_channel=num_filters) for _ in range(num_layers)]
        self.layers = nn.Sequential(
            *layers
        )

        self.last_conv = nn.Sequential(
            MaskedConvolution(mask_type='B', in_channels=num_filters, out_channels=num_filters, kernel_size=1),
            nn.ReLU(True),
            MaskedConvolution(mask_type='B', in_channels=num_filters, out_channels=num_filters, kernel_size=1),
            nn.ReLU(True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=num_filters, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.input_conv(inputs)
        x = self.layers(x)
        x = self.last_conv(x)
        out = self.out(x)
        return out


if __name__ == "__main__":
    A_mask_conv = MaskedConvolution('A', in_channels=1, out_channels=3, kernel_size=7, padding="same", bias=False)
    B_mask_conv = MaskedConvolution('B', in_channels=1, out_channels=3, kernel_size=7, padding="same", bias=False)

    print("A mask", A_mask_conv.mask)
    print("B mask", B_mask_conv.mask)
