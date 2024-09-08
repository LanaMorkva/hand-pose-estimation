import torch
import torch.nn as nn


class UNetDownsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDownsampling, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    

class UNetUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUpsampling, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x1 = torch.cat([x1, x2], dim=1)
        return self.double_conv(x1)
    

class UNet(nn.Module):
    def __init__(self, num_keypoints):
        super(UNet, self).__init__()

        # Define the encoder (U-Net downsampling part)
        self.encoder1 = UNetDownsampling(3, 32)
        self.encoder2 = UNetDownsampling(32, 64)
        self.encoder3 = UNetDownsampling(64, 128)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Define the decoder (U-Net upsampling part)
        self.upconv3 = UNetUpsampling(256, 128)
        self.upconv2 = UNetUpsampling(128, 64)
        self.upconv1 = UNetUpsampling(64, 32)
        
        # Final layer to produce heatmaps
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(32, num_keypoints, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Feature extraction
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.maxpool(x1))
        x3 = self.encoder3(self.maxpool(x2))

        x4 = self.middle(self.maxpool(x3))
        
        # Upsampling
        x = self.upconv3(x4, x3)
        x = self.upconv2(x, x2)
        x = self.upconv1(x, x1)
        
        # Output heatmaps
        heatmaps = self.heatmap_conv(x)
        return heatmaps