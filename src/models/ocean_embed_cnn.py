import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Use bilinear interpolation for arbitrary spatial dimensions, followed by a conv
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        # Pad if there are dimension mismatches
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
                        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class OceanEmbedCNN(nn.Module):
    def __init__(self, in_channels=7, out_channels=15, base_features=32):
        """
        U-Net style CNN for spatial ocean fields.
        Args:
            in_channels: Number of surface variables (7).
            out_channels: Number of depth levels (15).
            base_features: Number of filters in the first conv layer.
        """
        super(OceanEmbedCNN, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        
        # Encoder
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        
        # Bottleneck (Latent Representation / Satellite Embedding)
        self.down3 = Down(base_features * 4, base_features * 8)
        
        # Decoder
        self.up1 = Up(base_features * 12, base_features * 4)  # 8 + 4 = 12 channels from skip
        self.up2 = Up(base_features * 6, base_features * 2)   # 4 + 2 = 6
        self.up3 = Up(base_features * 3, base_features)       # 2 + 1 = 3
        
        self.outc = OutConv(base_features, out_channels)

    def forward(self, x):
        # x shape: (Batch, 7, Lat, Lon)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        embedding = self.down3(x3)  # This is the 'satellite embedding'
        
        x = self.up1(embedding, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        logits = self.outc(x)
        # logits shape: (Batch, 15, Lat, Lon)
        return logits
        
    def get_embedding(self, x):
        """Method to extract only the latent representation"""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        embedding = self.down3(x3)
        return embedding

if __name__ == "__main__":
    # Test the model with dummy input
    model = OceanEmbedCNN()
    # 100 lats, 240 lons as per 0.25 deg res for the region
    dummy_input = torch.randn(2, 7, 100, 240)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 15, 100, 240)
