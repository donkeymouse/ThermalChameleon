import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from thermal_embedding import ThermalMultiChannelEmbedding 
import numpy as np

class TCNet(nn.Module):
    """
    Adaptive channel-remapping using grouped convolutions, global pooling,
    and per-channel mixing weights computed via an MLP + softmax.
    """

    def __init__(self,
                 in_channels: int = 10,
                 out_channels: int = 3,
                 mlp_hidden: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = in_channels * out_channels

        # 1) Grouped conv layers: each input channel -> out_channels features
        self.conv1 = nn.Conv2d(
            in_channels,
            self.embed_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_channels
        )
        self.norm1 = nn.GroupNorm(num_groups=in_channels, num_channels=self.embed_dim)
        self.conv2 = nn.Conv2d(
            self.embed_dim,
            self.embed_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_channels
        )
        self.norm2 = nn.GroupNorm(num_groups=in_channels, num_channels=self.embed_dim)

        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 2) MLP for mixing weights
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.embed_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, self.embed_dim)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, H, W) or (N, H, W, C). Outputs (N, out_channels, H/4, W/4).
        """
        # accept NHWC by permuting
        if x.dim() == 4 and x.shape[1] != self.in_channels:
            x = x.permute(0, 3, 1, 2).contiguous()

        N, C, H, W = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        # feature extraction
        feat = self.relu(self.norm1(self.conv1(x)))   # (N, C*out, H/2, W/2)
        feat = self.relu(self.norm2(self.conv2(feat)))# (N, C*out, H/4, W/4)

        # global pooling -> flatten
        pooled = self.global_pool(feat)               # (N, embed_dim, 1, 1)
        pooled = pooled.view(N, self.embed_dim)       # (N, embed_dim)

        # compute mixing weights
        weights = self.mlp(pooled)                    # (N, embed_dim)
        weights = weights.view(N, C, self.out_channels)  # (N, C, out)
        weights = self.softmax(weights)               # softmax over C

        # weighted sum via torch.einsum -> (N, out, H, W)
        output = torch.einsum('nci,nchw->nihw', weights, x)
        return output




# Sanity check + example usage
if __name__ == '__main__':
    
    # Declare Thermal Chameleon
    _device = torch.device('cuda') #.device('cpu')
    img_path = './sample.png'
    raw_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if raw_img is None:
        raise FileNotFoundError(f"Cannot load image at {img_path}")
    
    # Convert to torch tensor of shape (1, H, W)
    raw_tensor = torch.from_numpy(raw_img).unsqueeze(0)

    # Initialize remapper and model
    remapper = ThermalMultiChannelEmbedding(
        num_output_channels=10,
        img_type='png',
        sync=True,
        device=_device
    )
    model = TCNet(in_channels=10, out_channels=3, mlp_hidden=64)
    model = model.to(_device)
    # Perform remapping
    remapped = remapper(raw_tensor)  # (1,10,H,W)
    print('Remapped shape:', remapped.shape)

    # Feed into TCNet
    output = model(remapped)
    print('TCNet output shape:', output.shape)

    