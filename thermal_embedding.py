import math
import torch
import torch.nn as nn


def convert_temp_tensor(raw: torch.Tensor) -> torch.Tensor:
    """
    Convert raw PNG sensor values to Celsius.

    Args:
        raw: Tensor of shape (..., H, W)
    Returns:
        Tensor of same shape, float32 Celsius
    """
    B = 1428.0  # calibration constant
    F = 1.0
    O = 118.126
    R = 377312.0
    T0 = 273.15
    # inside = R/(raw - O) + F
    inside = raw - O
    inside = torch.clamp(R / (inside + 1e-8) + F, min=1e-8)
    return B / torch.log(inside) - T0


def convert_flir_tensor(raw: torch.Tensor, kelvin: bool = False) -> torch.Tensor:
    """
    Convert raw FLIR TIFF values to Celsius (or Kelvin).

    Args:
        raw: Tensor of shape (..., H, W)
        kelvin: if True, return Kelvin scale
    Returns:
        Tensor of same shape
    """
    kelv = raw * 0.04
    return kelv if kelvin else (kelv - 273.15)


class ThermalMultiChannelEmbedding(nn.Module):
    """
    Multi-channel thermal embedding.
    Takes a batch tensor of single-channel thermal images and produces
    a stack of sinusoidally remapped channels.

    Input:
      x of shape (B, C, H, W) or (B, H, W), where C must be 1 if present.
    Output:
      Tensor of shape (B, num_output_channels, H, W), values in [0,1].
    """
    def __init__(self,
                 num_output_channels: int,
                 img_type: str = "png",
                 sync: bool = True,
                 device: torch.device = None):
        super().__init__()
        self.num_output_channels = num_output_channels
        self.img_type = img_type
        self.sync = sync
        self.device = device or torch.device('cpu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle NHWC or NCHW input
        print(x.shape)
        if x.dim() == 4:
            B, C, H, W = x.shape
            assert C == 1, f"Expected single-channel input, got C={C}"
            x = x.squeeze(1)  # -> (B, H, W)
        elif x.dim() == 3:
            B, H, W = x.shape
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")

        # Move to device and float
        temps = x.to(self.device).float()

        # Convert to temperature units
        if self.img_type.lower() == "png":
            temps = convert_temp_tensor(temps)
        elif self.img_type.lower() == "tiff":
            temps = convert_flir_tensor(temps)
        else:
            raise ValueError(f"Unsupported image type: {self.img_type}")

        # Generate D values: either shared across batch or per-sample
        if self.sync:
            D = torch.rand(self.num_output_channels, device=self.device) * (30 - 3) + 3
            D = D.unsqueeze(0).expand(B, -1)  # shape (B, num_output_channels)
        else:
            D = torch.rand(B, self.num_output_channels, device=self.device) * (30 - 3) + 3

        # Remap: divide and apply sinusoidal transform
        remap = temps.unsqueeze(1) / (D.view(B, self.num_output_channels, 1, 1) + 1e-8)
        remap = torch.sin(remap * (2 * math.pi / 6))  # range [-1,1]
        remap = remap * 0.5 + 0.5                    # range [0,1]
        remap = remap.clamp(0.0, 1.0)

        return remap

    __call__ = forward
