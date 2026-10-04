"""U-Net segmentation architectures: ResNet-34 backbone and Standard U-Net."""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.models.base import BaseSegmentationModel
from src.models.registry import register_model


class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DecoderBlock(nn.Module):
    """Upsampling followed by concatenation with skip connection and DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Handle slight dimension mismatch if odd dimensions
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


@register_model("resnet34_unet")
@register_model("unet")
class ResNet34UNet(BaseSegmentationModel):
    """U-Net with ImageNet pretrained ResNet-34 encoder and decoder with skip connections.

    Args:
        in_channels: Number of input image channels (default: 3 for RGB).
        out_channels: Number of output prediction channels (default: 1 for binary mask).
        pretrained: Whether to load ImageNet pretrained weights for the encoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)

        if in_channels != 3:
            # Adapt first conv if not 3 channels
            self.init_conv = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                resnet.bn1,
                resnet.relu,
            )
        else:
            self.init_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)

        self.maxpool = resnet.maxpool  # stride 2
        self.layer1 = resnet.layer1    # 64 ch, H/4
        self.layer2 = resnet.layer2    # 128 ch, H/8
        self.layer3 = resnet.layer3    # 256 ch, H/16
        self.layer4 = resnet.layer4    # 512 ch, H/32

        # Decoder stages
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)

        # Final upsample to match full input resolution H, W
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DoubleConv(32, 16),
        )
        self.head = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.init_conv(x)         # [B, 64, H/2, W/2]
        x1 = self.maxpool(x0)          # [B, 64, H/4, W/4]
        x1 = self.layer1(x1)           # [B, 64, H/4, W/4]
        x2 = self.layer2(x1)           # [B, 128, H/8, W/8]
        x3 = self.layer3(x2)           # [B, 256, H/16, W/16]
        x4 = self.layer4(x3)           # [B, 512, H/32, W/32]

        # Decoder with skip connections
        d4 = self.dec4(x4, x3)         # [B, 256, H/16, W/16]
        d3 = self.dec3(d4, x2)         # [B, 128, H/8, W/8]
        d2 = self.dec2(d3, x1)         # [B, 64, H/4, W/4]
        d1 = self.dec1(d2, x0)         # [B, 32, H/2, W/2]

        # Final upsampling
        out = self.final_up(d1)        # [B, 16, H, W]
        logits = self.head(out)        # [B, out_channels, H, W]
        return logits


@register_model("standard_unet")
class StandardUNet(BaseSegmentationModel):
    """Standard classical U-Net trained from scratch without pretrained weights."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down path
        curr_in = in_channels
        for feat in features:
            self.downs.append(DoubleConv(curr_in, feat))
            curr_in = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Up path
        rev_features = list(reversed(features))
        curr_in = features[-1] * 2
        for feat in rev_features:
            self.ups.append(nn.ConvTranspose2d(curr_in, feat, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feat * 2, feat))
            curr_in = feat

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([x, skip], dim=1)
            x = self.ups[i + 1](x)

        return self.head(x)
