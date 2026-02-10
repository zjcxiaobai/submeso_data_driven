import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Literal, Optional

# convolutional block
def conv_block(
    in_channels: int = 8,
    out_channels: int = 1, 
    kernel_size: int = 3, 
    activation: str = 'SiLU', 
    leakyrelu_hp: float = 0.2,
    batch_norm : Literal["bn","gn","in","none"] = "bn",
    order: str = "CNAD",
    dropout_2D: float = 0.0,
    dilation: int = 1,  # change the effective kernel size
    groups: int = 1,    # groups split channels into groups
    ):
    assert set(order).issubset({"C", "N", "A", "D"}), " order must use C/N/A/D" 
    assert kernel_size % 2 == 1, "odd kernel required with HybridPad2d padding"
    assert dilation >= 1, "dilation must be >= 1"
    pad = (kernel_size - 1) * dilation // 2 # stride = 1,
    has_norm = batch_norm != "none"
    
    layers: Dict[str, nn.Module] = {}
    
    layers["C"] = nn.Sequential(
    _HybridPad2d(pad_x=(pad), pad_y=(pad)),
    nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=0,  # no internal padding
        bias = (batch_norm == "none"), dilation=dilation, groups=groups
        )
    )
    layers["N"] = _make_norm(batch_norm, out_channels) if has_norm else nn.Identity()
    layers["A"] = _make_act(activation, leakyrelu_hp=leakyrelu_hp)
    layers["D"] = nn.Dropout2d(p=dropout_2D) if (dropout_2D and dropout_2D > 0.0) else nn.Identity()

    seq = [layers[k] for k in order if k in layers]
    return nn.Sequential(*seq)

#  choose activation
def _make_act(name: Optional[str], leakyrelu_hp) -> nn.Module:
    if not name or name.lower() in {"false", "none", "linear"}:
        return nn.Identity()
    name = name.lower()
    if name == "relu": return nn.ReLU(inplace = False)    # inplace tell python to modify the input tensor directly
    if name == "leakyrelu": return nn.LeakyReLU(negative_slope = leakyrelu_hp, inplace = False)
    if name == "silu": return nn.SiLU(inplace = False)
    raise ValueError(f"Unknown activation: {name}")
    
# batch normalization options
#  here apply the BN before activation, 
def _make_norm(kind: Literal["bn","gn","in","none"], num_channels: int,
               gn_groups: int = 16, bn_eps: float = 1e-3, bn_momentum: float = 0.1) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    if kind == "bn":
        return nn.BatchNorm2d(num_channels, eps=bn_eps, momentum=bn_momentum, affine=True)
    if kind == "in":
        return nn.InstanceNorm2d(num_channels, eps=1e-5, affine=True, track_running_stats=False)
    if kind == "gn":
        g = min(gn_groups, num_channels)
        while num_channels % g != 0:
            g -= 1
        return nn.GroupNorm(num_groups=g, num_channels=num_channels, eps=1e-5)
    raise ValueError(f"Unknown norm kind: {kind}")

# Hybrid padding
# padding, periodic in x and reflect in y
class _HybridPad2d(nn.Module):
    def __init__(self, pad_x: int, pad_y: int):
        super().__init__()
        self.pad_x = (pad_x, pad_x)
        self.pad_y = (pad_y, pad_y)
    def forward(self, x):
        # periodic in X
        x = nn.functional.pad(x, (self.pad_x[0], self.pad_x[1], 0, 0), mode="circular")
        # reflect in Y
        x = nn.functional.pad(x, (0, 0, self.pad_y[0], self.pad_y[1]), mode="reflect")
        return x
        
class DoubleConvPhys(nn.Module):
    """
    Two conv_blocks back-to-back.
    each block keeps H,W the same (for odd kernel sizes like 3).
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        activation: str = "SiLU",
        batch_norm: str = "bn",
        dropout_2D: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                activation=activation,
                batch_norm=batch_norm,
                order="CNAD",
                dropout_2D=dropout_2D,
            ),
            conv_block(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                activation=activation,
                batch_norm=batch_norm,
                order="CNAD",
                dropout_2D=dropout_2D,
            ),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downsample by 2 using pooling, then DoubleConvPhys.
    """
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(2) # pooling layer downsampling
        self.conv = DoubleConvPhys(in_ch, out_ch, **kwargs)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Upsample by 2, concat skip, then DoubleConvPhys.
    """
    def __init__(self, in_ch_from_down, in_ch_from_skip, out_ch, **kwargs):
        super().__init__()
        self.conv = DoubleConvPhys(in_ch_from_down + in_ch_from_skip, out_ch, **kwargs)

    def forward(self, x_down, x_skip):
        # Upsample to exactly match skip resolution
        x_down = F.interpolate(x_down, size=x_skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x_skip, x_down], dim=1)
        return self.conv(x)


class UNetPhysDepth1(nn.Module):
    """
    2-down / 2-up U-Net + 2 extra refinement conv layers at full resolution.
    Input : (B, input_channels, 88, 58)
    Output: (B, head_channel,  88, 58)
    """
    def __init__(
        self,
        input_channels: int = 5,
        head_channel: int = 1,
        base: int = 32,
        kernel_size: int = 3,
        activation: str = "SiLU",
        batch_norm: str = "bn",
        dropout_2D: float = 0.0,
    ):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConvPhys(
            input_channels, base,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
            dropout_2D=dropout_2D,
        )  # (B, base, 88, 58)

        self.down1 = Down(
            base, base * 2,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
            dropout_2D=dropout_2D,
        )  # (B, 2B, 44, 29)

        self.down2 = Down(
            base * 2, base * 4,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
            dropout_2D=dropout_2D,
        )  # (B, 4B, 22, 14)

        # Decoder
        self.up2 = Up(
            in_ch_from_down=base * 4,
            in_ch_from_skip=base * 2,
            out_ch=base * 2,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
            dropout_2D=dropout_2D,
        )  # (B, 2B, 44, 29)

        self.up1 = Up(
            in_ch_from_down=base * 2,
            in_ch_from_skip=base,
            out_ch=base,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
            dropout_2D=dropout_2D,
        )  # (B, B, 88, 58)

        # >>> NEW: two extra conv layers (same activation + norm)
        self.post = nn.Sequential(
            conv_block(
                in_channels=base,
                out_channels=base,
                kernel_size=kernel_size,
                activation=activation,
                batch_norm=batch_norm,
                order="CNAD",
                dropout_2D=dropout_2D,
            ),
            conv_block(
                in_channels=base,
                # out_channels=base,
                out_channels= int(base / 2),
                kernel_size=kernel_size,
                activation=activation,
                batch_norm=batch_norm,
                order="CNAD",
                dropout_2D=dropout_2D,
            ),
            conv_block(
                in_channels=int(base / 2),
                out_channels=int(base / 4),
                kernel_size=kernel_size,
                activation=activation,
                batch_norm=batch_norm,
                order="CNAD",
                dropout_2D=dropout_2D,
            ),
        )

        # Head
        self.head = nn.Conv2d(int(base / 4), head_channel, kernel_size=1, bias=True)

    def forward(self, x):
        x1 = self.enc1(x)      # (B, B, 88, 58)
        x2 = self.down1(x1)    # (B, 2B, 44, 29)
        x3 = self.down2(x2)    # (B, 4B, 22, 14)

        x  = self.up2(x3, x2)  # (B, 2B, 44, 29)
        x  = self.up1(x,  x1)  # (B, B,  88, 58)

        x  = self.post(x)      # >>> NEW refinement
        y  = self.head(x)
        return y
