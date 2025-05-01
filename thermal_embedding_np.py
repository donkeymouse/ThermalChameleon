import math
import numpy as np
import cv2
import argparse


def convert_temp_numpy(raw: np.ndarray) -> np.ndarray:
    """
    Convert raw PNG sensor values to Celsius.

    Args:
        raw: ndarray of shape (H, W)
    Returns:
        ndarray of same shape, float32 Celsius
    """
    B = 1428.0
    F = 1.0
    O = 118.126
    R = 377312.0
    T0 = 273.15
    inside = raw.astype(np.float32) - O
    inside = np.clip(R / (inside + 1e-8) + F, 1e-8, None)
    return B / np.log(inside) - T0


def convert_flir_numpy(raw: np.ndarray, kelvin: bool = False) -> np.ndarray:
    """
    Convert raw FLIR TIFF values to Celsius (or Kelvin).

    Args:
        raw: ndarray of shape (H, W)
        kelvin: if True, return Kelvin
    Returns:
        ndarray of same shape
    """
    kelv = raw.astype(np.float32) * 0.04
    return kelv if kelvin else (kelv - 273.15)


class ThermalMultiChannelEmbeddingNp:
    """
    Numpy-based multi-channel thermal embedding.
    Converts a single-channel thermal image to C-channel sinusoidal embeddings.

    Args:
        num_output_channels: number of remapped channels
        img_type: 'png' or 'tiff'
    """
    def __init__(self,
                 num_output_channels: int,
                 img_type: str = 'png'):
        self.C_out = num_output_channels
        self.img_type = img_type.lower()

    def embed(self,
              raw: np.ndarray,
              D_values: np.ndarray = None) -> np.ndarray:
        """
        raw: ndarray (H, W) or (H, W, 1)
        D_values: Optional 1D array of length C_out. If None, random in [3,30).
        Returns: ndarray (C_out, H, W) with values in [0,1]
        """
        # Ensure raw is 2D
        if raw.ndim == 3 and raw.shape[2] == 1:
            raw = raw[:, :, 0]
        elif raw.ndim != 2:
            raise ValueError(f"Unsupported raw shape {raw.shape}")

        # Convert to temperature
        if self.img_type == 'png':
            temps = convert_temp_numpy(raw)
        elif self.img_type == 'tiff':
            temps = convert_flir_numpy(raw)
        else:
            raise ValueError(f"Unsupported img_type {self.img_type}")

        H, W = temps.shape
        # Determine D values
        if D_values is None:
            D = np.random.uniform(3, 30, size=self.C_out).astype(np.float32)
        else:
            D = np.array(D_values, dtype=np.float32).reshape(-1)
            if D.shape[0] != self.C_out:
                raise ValueError(f"D_values length {D.shape[0]} != num_output_channels {self.C_out}")

        # Remap: divide and sinusoidal transform
        remap = temps[:, :, None] / (D[None, None, :] + 1e-8)   # (H, W, C_out)
        remap = np.sin(remap * (2 * math.pi / 6))              # [-1,1]
        remap = remap * 0.5 + 0.5                              # [0,1]
        remap = np.clip(remap, 0.0, 1.0)

        # Return as (C_out, H, W)
        return remap.transpose(2, 0, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo: Numpy-based thermal embedding with tiled OpenCV visualization'
    )
    parser.add_argument('img_path', type=str, help='Path to raw thermal image')
    parser.add_argument('--num_channels', type=int, default=10,
                        help='Number of remapped channels')
    parser.add_argument('--img_type', type=str, choices=['png','tiff'], default='png',
                        help='Input image type')
    parser.add_argument('--d_values', type=float, nargs='+',
                        help='Optional list of D values (length=num_channels)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for D sampling')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Load image
    raw = cv2.imread(args.img_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot load image at {args.img_path}")
    if raw.ndim == 3:
        raw = raw[:, :, 0]

    # Determine D values
    if args.d_values:
        D_values = np.array(args.d_values, dtype=np.float32)
        if D_values.shape[0] != args.num_channels:
            raise ValueError("--d_values length must equal num_channels")
    else:
        D_values = np.random.uniform(3, 30, size=args.num_channels).astype(np.float32)

    # Embed
    embedder = ThermalMultiChannelEmbeddingNp(
        num_output_channels=args.num_channels,
        img_type=args.img_type
    )
    out = embedder.embed(raw, D_values=D_values)

    C, H, W = out.shape
    # Create color mosaic for all channels and overlay D
    cols = int(math.ceil(math.sqrt(C)))
    rows = int(math.ceil(C / cols))
    mosaic = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
    for idx in range(C):
        i = idx // cols
        j = idx % cols
        # Channel image -> BGR
        channel_gray = (out[idx] * 255).astype(np.uint8)
        channel_bgr = cv2.cvtColor(channel_gray, cv2.COLOR_GRAY2BGR)
        # Overlay D value in blue (BGR)
        text = f"D={D_values[idx]:.1f}"
        cv2.putText(channel_bgr, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 3, lineType=cv2.LINE_AA)
        # Place in mosaic
        mosaic[i*H:(i+1)*H, j*W:(j+1)*W] = channel_bgr

    # Show mosaic in one window
    window_name = 'Thermal Channels'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
