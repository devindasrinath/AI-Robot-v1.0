"""
DS-CNN Wake Word Detector for "Nexu"

Architecture: Depthwise Separable CNN on log-mel spectrogram
Input:  [1, 1, 40, 98]  — 1ch, 40 mel bins, 98 frames (~1 second at 10ms hop)
Output: [1, 2]          — [no_wake, wake] logits

All ops: Conv2d, DepthwiseConv2d (groups=C), BatchNorm, ReLU, GlobalAvgPool, Linear
→ all safe for tpu-mlir CV181x compilation.
"""

import torch
import torch.nn as nn


class DSConvBlock(nn.Module):
    """Depthwise separable conv: depthwise + pointwise + BN + ReLU."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                            padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.relu(x)


class WakeWordCNN(nn.Module):
    """
    DS-CNN for single keyword spotting ("Nexu").

    Input shape:  [batch, 1, 40, 98]
    Output shape: [batch, 2]   logits — [no_wake, wake]

    ~22K parameters — fits easily in CV181x NPU SRAM.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Initial standard conv to lift from 1 channel to 32
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # DS-CNN body: 4 depthwise separable blocks
        self.ds_blocks = nn.Sequential(
            DSConvBlock(32, 32, stride=2),   # → [32, 20, 49]
            DSConvBlock(32, 64, stride=2),   # → [64, 10, 25]
            DSConvBlock(64, 64, stride=2),   # → [64,  5, 13]
            DSConvBlock(64, 64, stride=1),   # → [64,  5, 13]
        )

        # Fixed average pool → [batch, 64]
        # Feature map after DS blocks: [64, 5, 13] — use exact kernel to avoid
        # AdaptiveAvgPool2d which exports as ReduceMean (unsupported in tpu-mlir)
        self.gap = nn.AvgPool2d(kernel_size=(5, 13))

        # Classifier head
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, 1, 40, 98]"""
        x = self.stem(x)
        x = self.ds_blocks(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = WakeWordCNN()
    model.eval()

    dummy = torch.randn(1, 1, 40, 98)
    out = model(dummy)

    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}")

    # Verify all output dims
    assert out.shape == (1, 2), f"Expected (1,2), got {out.shape}"
    print("Architecture OK")
