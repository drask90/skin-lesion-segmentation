# -*- coding: utf-8 -*-
"""
Focal Modulation U-Net (PyTorch) — aligned with the TensorFlow version.
"""

import torch
import torch.nn as nn


class FocalModulationBlock(nn.Module):
    """
    Matches the TensorFlow focal_modulation_block:
      1. Channel-wise mean (GAP) and max (GMP)
      2. modulation = (max - mean) * alpha
      3. 1x1 Conv + Sigmoid
      4. Scale input by modulation
      5. Raise to power gamma
    """
    def __init__(self, dim, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise mean (Global Average Pooling)
        mean = x.mean(dim=(2, 3), keepdim=True)          # (B, C, 1, 1)
        # Channel-wise max (Global Max Pooling)
        max_val = x.amax(dim=(2, 3), keepdim=True)       # (B, C, 1, 1)
        # Modulation factor
        modulation = (max_val - mean) * self.alpha        # (B, C, 1, 1)
        modulation = self.sigmoid(self.conv(modulation))  # (B, C, 1, 1)
        # Scale and apply gamma
        scaled = x * modulation
        outputs = scaled ** self.gamma
        return outputs


class FocalModulationContextAggregation(nn.Module):
    """
    Matches the TensorFlow focal_modulation_context_aggregation_block:
      1. 3x3 Conv (local context)  → conv1
      2. 1x1 Conv (global context) → conv2 → GAP → 1x1 Conv + Sigmoid → multiply with conv1
      3. FocalModulationBlock on the result
      4. Concatenate conv1 and focal modulation output → output has 2x channels
    """
    def __init__(self, in_ch, mid_ch=32, gamma=2.0, alpha=0.25):
        super().__init__()
        # Local context: 3x3 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Global context: 1x1 conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        # Global context gate: 1x1 conv + sigmoid
        self.global_gate = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        # Focal modulation block
        self.focal = FocalModulationBlock(mid_ch, gamma=gamma, alpha=alpha)

    def forward(self, x):
        # Local context
        conv1 = self.conv1(x)                                   # (B, mid_ch, H, W)
        # Global context
        conv2 = self.conv2(x)                                   # (B, mid_ch, H, W)
        global_ctx = conv2.mean(dim=(2, 3), keepdim=True)       # (B, mid_ch, 1, 1)
        global_ctx = self.global_gate(global_ctx)                # (B, mid_ch, 1, 1)
        global_context = conv1 * global_ctx                     # (B, mid_ch, H, W)
        # Focal modulation
        focal_out = self.focal(global_context)                  # (B, mid_ch, H, W)
        # Concatenate local context and focal modulation output
        out = torch.cat([conv1, focal_out], dim=1)              # (B, 2*mid_ch, H, W)
        return out


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, base=16):
        super().__init__()
        # ---------- Encoder ----------
        self.enc1 = _DoubleConv(in_ch, base)
        self.enc2 = _DoubleConv(base,     base * 2)
        self.enc3 = _DoubleConv(base * 2, base * 4)
        self.enc4 = _DoubleConv(base * 4, base * 8)

        # ---------- Bottleneck ----------
        self.bot = _DoubleConv(base * 8, base * 16)
        # FocalModulationContextAggregation outputs 2*mid_ch,
        # so we set mid_ch = base*8 to get base*16 back after concat.
        self.bot_fm = FocalModulationContextAggregation(base * 16, mid_ch=base * 8)

        # ---------- Decoder ----------
        self.up4  = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.fm4  = FocalModulationContextAggregation(base * 8, mid_ch=base * 4)
        self.dec4 = _DoubleConv(base * 16, base * 8)

        self.up3  = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.fm3  = FocalModulationContextAggregation(base * 4, mid_ch=base * 2)
        self.dec3 = _DoubleConv(base * 8, base * 4)

        self.up2  = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.fm2  = FocalModulationContextAggregation(base * 2, mid_ch=base)
        self.dec2 = _DoubleConv(base * 4, base * 2)

        self.up1  = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.fm1  = FocalModulationContextAggregation(base, mid_ch=base // 2)
        self.dec1 = _DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bot_fm(self.bot(self.pool(e4)))

        # Decoder with focal-modulated skip connections
        up4 = self.up4(b)
        d4  = self.dec4(torch.cat([up4, self.fm4(e4)], dim=1))

        up3 = self.up3(d4)
        d3  = self.dec3(torch.cat([up3, self.fm3(e3)], dim=1))

        up2 = self.up2(d3)
        d2  = self.dec2(torch.cat([up2, self.fm2(e2)], dim=1))

        up1 = self.up1(d2)
        d1  = self.dec1(torch.cat([up1, self.fm1(e1)], dim=1))

        return self.head(d1)


if __name__ == '__main__':
    model = UNet()
    dummy_input = torch.randn(1, 3, 256, 256)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params / 1e6:.2f}M")
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        print(f"FLOPs:      {macs * 2 / 1e9:.2f}G")
    except ImportError:
        print("FLOPs:      install 'thop' to compute  (pip install thop)")

    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape:  {dummy_input.shape}")
        print(f"Output shape: {output.shape}")