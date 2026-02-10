import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    """
    ConvLSTM cell.
    Input:  x_t  (B, Cin, H, W)
    State: (h, c) both (B, Chid, H, W)
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3, bias=True):
        super().__init__()
        assert kernel_size % 2 == 1, "Use odd kernel so padding keeps H,W"
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # One conv produces all 4 gates at once
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, state):
        h, c = state  # (B, Chid, H, W)
        cat = torch.cat([x, h], dim=1)  # (B, Cin+Chid, H, W)
        gates = self.conv(cat)          # (B, 4*Chid, H, W)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch_size, spatial_size, device, dtype):
        H, W = spatial_size
        h = torch.zeros(batch_size, self.hidden_channels, H, W, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_channels, H, W, device=device, dtype=dtype)
        return h, c


class BottleneckConvLSTM(nn.Module):
    """
    Runs ConvLSTM over time on bottleneck feature maps.
    Input:  x3_seq (B, T, Cb, Hb, Wb)
    Output: h_last (B, Chid, Hb, Wb)   <-- rewritten bottleneck features
    """
    def __init__(self, in_channels, hidden_channels=None, kernel_size=3):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels  # common default: keep same channels
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size=kernel_size)

    def forward(self, x3_seq):
        B, T, C, H, W = x3_seq.shape
        device = x3_seq.device
        dtype = x3_seq.dtype

        h, c = self.cell.init_state(B, (H, W), device=device, dtype=dtype)

        for t in range(T):
            x_t = x3_seq[:, t]  # (B, C, H, W)
            h, c = self.cell(x_t, (h, c))

        return h  # last hidden state


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
    Minimal U-Net (2 down, 2 up) using your conv_block everywhere.
    Input : (B, input_channels, 88, 58) OR (B, T, input_channels, 88, 58)
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

        # --- LSTM knobs (safe defaults) ---
        use_lstm: bool = True,
        lstm_in_dim: int = 256,
        lstm_hidden: int = 512,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
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

        # --- LSTM at bottleneck (optional) ---
        self.use_lstm = use_lstm
        if use_lstm:
            self.mid_lstm = BottleneckConvLSTM(
                in_channels=base * 4,
                hidden_channels=base * 4,   # keep same channels
                kernel_size=3,
            )
        self.lstm_dropout = nn.Dropout2d(p=lstm_dropout) if (use_lstm and lstm_dropout > 0) else nn.Identity()


        # Decoder
        self.up2 = Up(
            in_ch_from_down=base * 4,
            in_ch_from_skip=base * 2,
            out_ch=base * 2,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
            dropout_2D=dropout_2D,
        )

        self.up1 = Up(
            in_ch_from_down=base * 2,
            in_ch_from_skip=base,
            out_ch=base,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
            dropout_2D=dropout_2D,
        )

        self.head = nn.Conv2d(base, head_channel, kernel_size=1, bias=True)

    def forward(self, x):
        # Backward compatible: accept (B,C,H,W) or (B,T,C,H,W)
        if x.dim() == 4:
            # Original behavior (no temporal modeling)
            x1 = self.enc1(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)

            x  = self.up2(x3, x2)
            x  = self.up1(x,  x1)
            y  = self.head(x)
            return y

        if x.dim() != 5:
            raise ValueError(f"Expected x dim 4 or 5, got {x.dim()} with shape {tuple(x.shape)}")

        # Sequence path: x is (B, T, C, H, W)
        B, T, C, H, W = x.shape

        x1_list, x2_list, x3_list = [], [], []
        for t in range(T):
            xt = x[:, t, :, :, :]     # (B, C, H, W)
            x1 = self.enc1(xt)        # (B, B, 88, 58)
            x2 = self.down1(x1)       # (B, 2B, 44, 29)
            x3 = self.down2(x2)       # (B, 4B, 22, 14)
            x1_list.append(x1)
            x2_list.append(x2)
            x3_list.append(x3)

        x1_last = x1_list[-1]
        x2_last = x2_list[-1]
        x3_seq  = torch.stack(x3_list, dim=1)  # (B, T, 4B, 22, 14)
        
        if self.use_lstm:
            x3_last = self.mid_lstm(x3_seq)      # (B, 4B, Hb, Wb)
            x3_last = self.lstm_dropout(x3_last) # ✅ apply dropout here
        else:
            x3_last = x3_seq[:, -1]



        # Decode last step (keeps your skip structure)
        x = self.up2(x3_last, x2_last)
        x = self.up1(x, x1_last)
        y = self.head(x)
        return y
