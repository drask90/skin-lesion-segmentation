# -*- coding: utf-8 -*-
"""
Bidirectional ConvLSTM U-Net (PyTorch).
Skip connections are processed with a row-wise Bidirectional ConvLSTM that
scans the feature map forward (row 0→H-1) and backward (row H-1→0),
concatenates both directions, and projects back to the original channel count
via a residual 1×1 conv.
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Bidirectional ConvLSTM components
# ─────────────────────────────────────────────────────────────────────────────

class ConvLSTMCell(nn.Module):
    """Standard 2-D Convolutional LSTM cell (i, f, g, o gates in one conv)."""

    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        pad = kernel_size // 2
        # Single conv for all four gates over (input ∥ hidden)
        self.conv = nn.Conv2d(
            in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad, bias=True
        )

    def forward(self, x, h, c):
        """
        x : (B, in_ch,     1, W)   — one row of the feature map
        h : (B, hidden_ch, 1, W)   — hidden state
        c : (B, hidden_ch, 1, W)   — cell state
        """
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = gates.chunk(4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, B, W, device):
        z = torch.zeros(B, self.hidden_ch, 1, W, device=device)
        return z, z.clone()


class BidirectionalConvLSTMSkip(nn.Module):
    """
    Bidirectional ConvLSTM applied to an encoder skip-connection feature map.

    The spatial H dimension is treated as a sequence:
      • Forward  cell: scans rows 0 → H-1
      • Backward cell: scans rows H-1 → 0 (outputs re-aligned to 0…H-1)

    The two hidden-state sequences are concatenated (2·hidden_ch channels) and
    projected back to in_ch with a 1×1 conv + BN + ReLU, then added to the
    input as a residual connection.

    Output shape == input shape.
    """

    def __init__(self, in_ch, hidden_ch):
        super().__init__()
        self.fwd_cell = ConvLSTMCell(in_ch, hidden_ch)
        self.bwd_cell = ConvLSTMCell(in_ch, hidden_ch)
        self.proj = nn.Sequential(
            nn.Conv2d(2 * hidden_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # ── Forward pass (row 0 → H-1) ──────────────────────────────────────
        h_f, c_f = self.fwd_cell.init_hidden(B, W, x.device)
        fwd_out = []
        for i in range(H):
            h_f, c_f = self.fwd_cell(x[:, :, i : i + 1, :], h_f, c_f)
            fwd_out.append(h_f)

        # ── Backward pass (row H-1 → 0, stored in original order) ───────────
        h_b, c_b = self.bwd_cell.init_hidden(B, W, x.device)
        bwd_out = [None] * H
        for i in range(H - 1, -1, -1):
            h_b, c_b = self.bwd_cell(x[:, :, i : i + 1, :], h_b, c_b)
            bwd_out[i] = h_b

        # ── Assemble, project, residual ──────────────────────────────────────
        fwd = torch.cat(fwd_out, dim=2)           # (B, hidden_ch, H, W)
        bwd = torch.cat(bwd_out, dim=2)           # (B, hidden_ch, H, W)
        out = self.proj(torch.cat([fwd, bwd], dim=1))  # (B, in_ch, H, W)
        return out + x                            # residual


# ─────────────────────────────────────────────────────────────────────────────
# Shared building block
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Lightweight U-Net with Bidirectional ConvLSTM skip connections.

    Encoder feature maps (e1…e4) each pass through a BidirectionalConvLSTMSkip
    before being concatenated with the upsampled decoder path.
    Channel counts are preserved so all existing decoder dimensions stay intact.

    Layer channel widths (base=16):
        enc1 →  16   bclstm1: hidden= 8
        enc2 →  32   bclstm2: hidden=16
        enc3 →  64   bclstm3: hidden=32
        enc4 → 128   bclstm4: hidden=64
        bot  → 256
    """

    def __init__(self, in_ch=3, base=16):
        super().__init__()

        # ── Encoder ─────────────────────────────────────────────────────────
        self.enc1 = _DoubleConv(in_ch,      base)
        self.enc2 = _DoubleConv(base,       base * 2)
        self.enc3 = _DoubleConv(base * 2,   base * 4)
        self.enc4 = _DoubleConv(base * 4,   base * 8)

        # ── Bottleneck (plain double-conv, no attention) ─────────────────────
        self.bot  = _DoubleConv(base * 8,   base * 16)

        # ── Bidirectional ConvLSTM on each skip connection ───────────────────
        self.bclstm4 = BidirectionalConvLSTMSkip(base * 8, hidden_ch=base * 4)
        self.bclstm3 = BidirectionalConvLSTMSkip(base * 4, hidden_ch=base * 2)
        self.bclstm2 = BidirectionalConvLSTMSkip(base * 2, hidden_ch=base)
        self.bclstm1 = BidirectionalConvLSTMSkip(base,     hidden_ch=base // 2)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up4  = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.dec4 = _DoubleConv(base * 16, base * 8)   # up(128) + skip(128)

        self.up3  = nn.ConvTranspose2d(base * 8,  base * 4, 2, 2)
        self.dec3 = _DoubleConv(base * 8,  base * 4)   # up(64)  + skip(64)

        self.up2  = nn.ConvTranspose2d(base * 4,  base * 2, 2, 2)
        self.dec2 = _DoubleConv(base * 4,  base * 2)   # up(32)  + skip(32)

        self.up1  = nn.ConvTranspose2d(base * 2,  base,     2, 2)
        self.dec1 = _DoubleConv(base * 2,  base)        # up(16)  + skip(16)

        self.head = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bot(self.pool(e4))

        # Decoder — BiConvLSTM-enhanced skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  self.bclstm4(e4)], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), self.bclstm3(e3)], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.bclstm2(e2)], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.bclstm1(e1)], dim=1))

        return self.head(d1)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

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
