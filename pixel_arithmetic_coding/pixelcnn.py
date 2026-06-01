"""
pixelcnn.py
-----------
PixelCNN model for Fashion MNIST autoregressive pixel modelling.

Architecture follows van den Oord et al., "Pixel Recurrent Neural Networks" (2016)
with residual blocks from "Conditional Image Generation with PixelCNN Decoders".

Key property that distinguishes this from GPT-2 in the text experiment:
  - ENCODING: one forward pass gives P(x_i | x_<i) for ALL pixels simultaneously.
    Masked convolutions enforce the causal constraint across the whole image at once.
  - DECODING: 784 sequential forward passes (pixel must exist before it can condition the next).

Input : (B, 1, 28, 28)  uint8 pixel values [0, 255]
Output: (B, 256, 28, 28) logits — one 256-way categorical per pixel position
Loss  : cross_entropy(output, target) where target is (B, 28, 28) int64
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Masked convolution
# ---------------------------------------------------------------------------

class MaskedConv2d(nn.Conv2d):
    """
    Standard Conv2d with a binary causal mask applied to the weights.

    Mask type 'A': current pixel is NOT included → used only in the first layer.
    Mask type 'B': current pixel IS included     → used in all subsequent layers.

    The mask zeros out all weight connections that would allow information from
    pixels later in raster order (row-major) to influence the current prediction.

    For a kernel of height kH and width kW, centre at (kH//2, kW//2):
      - All rows above centre:             1  (past pixels, allowed)
      - Centre row, left of centre:        1  (past pixels, allowed)
      - Centre pixel itself:               1 if type B, 0 if type A
      - Centre row, right of centre:       0  (future pixels, blocked)
      - All rows below centre:             0  (future pixels, blocked)
    """

    def __init__(self, mask_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ("A", "B"), "mask_type must be 'A' or 'B'"
        _, _, kH, kW = self.weight.shape
        mask = torch.ones_like(self.weight)
        # Block everything after (and for type A, including) the centre pixel
        mask[:, :, kH // 2, kW // 2 + (1 if mask_type == "B" else 0):] = 0
        mask[:, :, kH // 2 + 1:] = 0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Bottleneck residual block using type-B masked convolutions.

    Structure:  ReLU → 1×1 conv (ch → ch/2)
                ReLU → 3×3 masked conv (ch/2 → ch/2)
                ReLU → 1×1 conv (ch/2 → ch)
                + skip connection

    The 1×1 convolutions are also masked (type B) so the causal constraint
    is preserved throughout the residual path.
    """

    def __init__(self, channels: int):
        super().__init__()
        half = channels // 2
        self.block = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d("B", channels, half, kernel_size=1),
            nn.ReLU(),
            MaskedConv2d("B", half, half, kernel_size=3, padding=1),
            nn.ReLU(),
            MaskedConv2d("B", half, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# PixelCNN
# ---------------------------------------------------------------------------

class PixelCNN(nn.Module):
    """
    PixelCNN for 28×28 single-channel (grayscale) images.

    Parameters
    ----------
    n_channels : int
        Number of feature channels throughout the network.
        64 trains in ~15 min on a T4 and achieves ~3.6–3.8 BPP on Fashion MNIST.
        128 gives ~0.1–0.2 BPP improvement at ~2× training time.
    n_residual_blocks : int
        Depth of the residual stack. 8 is a good default.
    """

    def __init__(self, n_channels: int = 64, n_residual_blocks: int = 8):
        super().__init__()

        # Type-A mask: first layer must NOT see the pixel it is predicting
        self.input_conv = MaskedConv2d("A", 1, n_channels, kernel_size=7, padding=3)

        # Residual stack with type-B masks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(n_channels) for _ in range(n_residual_blocks)]
        )

        # Output head: two 1×1 convolutions → 256 logits per pixel
        self.output_conv = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d("B", n_channels, n_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(n_channels, 256, kernel_size=1),   # final conv need not be masked
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, 28, 28) integer pixel values [0, 255]
        returns: (B, 256, 28, 28) logits
        """
        x = x.float() / 255.0 - 0.5          # normalise to [-0.5, 0.5]
        x = self.input_conv(x)
        x = self.residual_blocks(x)
        return self.output_conv(x)
