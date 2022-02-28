import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels= 3, out_channels=1, channel_ind=[3, 64, 128, 256, 512, 1024]):
        super(UNet, self).__init__()

        channel_ind[0] = in_channels
        self.contraction = nn.ModuleList()
        self.expansive = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Contraction, i.e. downsampling
        for i in range(len(channel_ind) - 2):
            self.contraction.append(DoubleConvReLU(channel_ind[i], channel_ind[i + 1]))

        self.bottom = DoubleConvReLU(channel_ind[-2], channel_ind[-1])

        # Expansive, i.e. upsampling
        for i in range(len(channel_ind) - 1, 1, -1):
            self.expansive.append(nn.ConvTranspose2d(channel_ind[i], channel_ind[i - 1], kernel_size=2, stride=2))
            self.expansive.append(DoubleConvReLU(channel_ind[i], channel_ind[i - 1]))

        self.conv1 = nn.Conv2d(channel_ind[1], out_channels, kernel_size=1)

    def forward(self, x):
        skip = []

        for e in self.contraction:
            x = e(x)
            skip.append(x)
            x = self.pool(x)

        x = self.bottom(x)

        skip = skip[::-1] #  reverese list
        for i in range(0, len(self.expansive), 2):
            x = self.expansive[i](x)

            connection = skip[i // 2]
            # In case the max pool floors the width/height
            if connection.shape != x.shape:
                x = TF.resize(x, size=connection.shape[2:])

            x = torch.cat((connection, x), dim=1)
            x = self.expansive[i + 1](x)
        x = self.conv1(x)
        return x