import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Literal, Optional

class ConvLSTMCell(nn.Module):
    """
    ConvLSTM cell with Hybrid padding (periodic x, reflect y).
    Input:  x_t  (B, Cin, H, W)
    State: (h, c) both (B, Chid, H, W)
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3, bias=True):
        super().__init__()
        assert kernel_size % 2 == 1, "Use odd kernel so padding keeps H,W"
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2

        # Hybrid pad BEFORE the conv, so conv uses padding=0
        self.pad = _HybridPad2d(pad_x=pad, pad_y=pad)

        # One conv produces all 4 gates at once (no internal padding)
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=0,
            bias=bias,
        )

    def forward(self, x, state):
        h, c = state  # (B, Chid, H, W)
        cat = torch.cat([x, h], dim=1)        # (B, Cin+Chid, H, W)
        cat = self.pad(cat)                   # hybrid padded
        gates = self.conv(cat)                # (B, 4*Chid, H, W)

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
    Input:  x_seq (B, T, Cb, Hb, Wb)
    Output: h_last (B, Chid, Hb, Wb)   <-- rewritten bottleneck features
    """
    def __init__(self, in_channels, hidden_channels=None, kernel_size=3):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels  # common default: keep same channels
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size=kernel_size)

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        device = x_seq.device
        dtype = x_seq.dtype
    
        h, c = self.cell.init_state(B, (H, W), device=device, dtype=dtype)
    
        for t in range(T):
            x_t = x_seq[:, t]
            h, c = self.cell(x_t, (h, c))
    
        return h


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



class UpPixelShuffle(nn.Module):
    """
    Upsample by 2 using PixelShuffle, concat skip, then DoubleConvPhys.

    Uses HybridPad2d before the upsampling conv so spatial sizes are preserved
    and boundary conditions are consistent with the rest of the network.
    """
    def __init__(self, in_ch_from_down: int, in_ch_from_skip: int, out_ch: int, **kwargs):
        super().__init__()
        r = 2  # upsample factor

        # (B, in_ch_from_down, H, W)
        # -> HybridPad(1) + Conv3x3(no internal padding) keeps H,W
        # -> (B, out_ch*r*r, H, W)
        # -> PixelShuffle -> (B, out_ch, 2H, 2W)
        self.up = nn.Sequential(
            _HybridPad2d(pad_x=1, pad_y=1),
            nn.Conv2d(in_ch_from_down, out_ch * (r * r), kernel_size=3, padding=0, bias=True),
            nn.PixelShuffle(r),
        )

        # after concat: (out_ch + in_ch_from_skip) -> out_ch
        # NOTE: DoubleConvPhys is assumed to already use conv_block -> _HybridPad2d internally
        self.conv = DoubleConvPhys(out_ch + in_ch_from_skip, out_ch, **kwargs)

    def forward(self, x_down: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_down = self.up(x_down)

        # safety: ensure exact match (handles odd shapes / pooling rounding)
        if x_down.shape[-2:] != x_skip.shape[-2:]:
            x_down = F.interpolate(x_down, size=x_skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x_skip, x_down], dim=1)
        return self.conv(x)


class UNetPhysDepth1(nn.Module):
    """
    Minimal U-Net (3 down, 3 up) using your conv_block everywhere.
    Input : (B, input_channels, 88, 58) OR (B, T, input_channels, 88, 58)
    Output: (B, head_channel,  88, 58)
    """
    def __init__(
        self,
        input_channels: int = 7,
        head_channel: int = 1,
        base: int = 32,
        kernel_size: int = 3,
        activation: str = "SiLU",
        batch_norm: str = "bn",
        dropout_2D: float = 0.0,

        # --- LSTM knobs (safe defaults) ---
        use_lstm: bool = True,
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
        self.down3 = Down(
            base * 4, base * 8,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
            dropout_2D=dropout_2D,
        )  # (B, 8B, 11, 7)

        # --- LSTM at bottleneck (optional) ---
        self.use_lstm = use_lstm
        if use_lstm:
            self.mid_lstm = BottleneckConvLSTM(
                in_channels=base * 8,
                hidden_channels=base * 8,   # keep same channels
                kernel_size=3,
            )
        self.lstm_dropout = nn.Dropout2d(p=lstm_dropout) if (use_lstm and lstm_dropout > 0) else nn.Identity()


        # Decoder
        self.up3 = UpPixelShuffle(
            in_ch_from_down=base * 8,
            in_ch_from_skip=base * 4,
            out_ch=base * 4,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
            dropout_2D=dropout_2D,
        )

        self.up2 = UpPixelShuffle(
            in_ch_from_down=base * 4,
            in_ch_from_skip=base * 2,
            out_ch=base * 2,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
            dropout_2D=dropout_2D,
        )

        self.up1 = UpPixelShuffle(
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
        # Accept (B,C,H,W) or (B,T,C,H,W)
        if x.dim() == 4:
            x1 = self.enc1(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
    
            x  = self.up3(x4, x3)
            x  = self.up2(x,  x2)
            x  = self.up1(x,  x1)
            y  = self.head(x)
            return y
    
        if x.dim() != 5:
            raise ValueError(f"Expected x dim 4 or 5, got {x.dim()} with shape {tuple(x.shape)}")
    
        # Sequence path: x is (B, T, C, H, W)
        B, T, C, H, W = x.shape
    
        x4_list = []
        x1_last = x2_last = x3_last = None
    
        for t in range(T):
            xt = x[:, t]
            x1 = self.enc1(xt)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x4_list.append(x4)
            x1_last, x2_last, x3_last = x1, x2, x3
    
        x4_seq = torch.stack(x4_list, dim=1)  # (B,T,8B,Hb,Wb)
    
        if self.use_lstm:
            x4_last = self.mid_lstm(x4_seq)
            x4_last = self.lstm_dropout(x4_last)
        else:
            x4_last = x4_seq[:, -1]
    
        x = self.up3(x4_last, x3_last)
        x = self.up2(x, x2_last)
        x = self.up1(x, x1_last)
        y = self.head(x)
        return y
