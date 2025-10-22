import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import math


# CBAM Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(F.adaptive_avg_pool2d(x, 1))))
        max_out = self.fc2(self.relu1(self.fc1(F.adaptive_max_pool2d(x, 1))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# Transformer Fusion Module
class TransformerFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=1024,
                                                   dropout=0.1,
                                                   activation='relu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, features):
        # features: list of [B, C] tensors → stack → [B, N, C]
        x = torch.stack(features, dim=1)
        x = self.encoder(x)  # [B, N, C]
        return x.mean(dim=1) # global fused vector


# WAFFNet++ with CBAM + Transformer fusion
class WAFFNetPP(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        base = resnet50(pretrained=True)

        self.stage1 = nn.Sequential(*list(base.children())[:5])   # C2, 256 channels
        self.stage2 = nn.Sequential(*list(base.children())[5:6])  # C3, 512 channels
        self.stage3 = nn.Sequential(*list(base.children())[6:7])  # C4, 1024 channels

        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)

        self.proj1 = nn.Linear(256, 512)
        self.proj2 = nn.Linear(512, 512)
        self.proj3 = nn.Linear(1024, 512)

        self.fusion = TransformerFusion(embed_dim=512, num_heads=4, num_layers=2)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f1 = self.cbam1(self.stage1(x))
        f2 = self.cbam2(self.stage2(f1))
        f3 = self.cbam3(self.stage3(f2))

        f1 = F.adaptive_avg_pool2d(f1, 1).view(x.size(0), -1)
        f2 = F.adaptive_avg_pool2d(f2, 1).view(x.size(0), -1)
        f3 = F.adaptive_avg_pool2d(f3, 1).view(x.size(0), -1)

        f1 = self.proj1(f1)
        f2 = self.proj2(f2)
        f3 = self.proj3(f3)

        fused = self.fusion([f1, f2, f3])
        return self.fc(fused)