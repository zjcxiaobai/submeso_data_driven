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

        self.pad = _HybridPad2d(pad_x=pad, pad_y=pad)

        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=0,
            bias=bias,
        )

    def forward(self, x, state):
        h, c = state
        cat = torch.cat([x, h], dim=1)
        cat = self.pad(cat)
        gates = self.conv(cat)

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
    Output: h_last (B, Chid, Hb, Wb)
    """
    def __init__(self, in_channels, hidden_channels=None, kernel_size=3):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
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


def conv_block(
    in_channels: int = 8,
    out_channels: int = 1,
    kernel_size: int = 3,
    activation: str = 'SiLU',
    leakyrelu_hp: float = 0.2,
    batch_norm: Literal["bn", "gn", "in", "none"] = "bn",
    order: str = "CNAD",
    dropout_2D: float = 0.0,
    dilation: int = 1,
    groups: int = 1,
):
    assert set(order).issubset({"C", "N", "A", "D"}), "order must use C/N/A/D"
    assert kernel_size % 2 == 1, "odd kernel required with HybridPad2d padding"
    assert dilation >= 1, "dilation must be >= 1"

    pad = (kernel_size - 1) * dilation // 2
    has_norm = batch_norm != "none"

    layers: Dict[str, nn.Module] = {}

    layers["C"] = nn.Sequential(
        _HybridPad2d(pad_x=pad, pad_y=pad),
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            bias=(batch_norm == "none"),
            dilation=dilation,
            groups=groups,
        ),
    )
    layers["N"] = _make_norm(batch_norm, out_channels) if has_norm else nn.Identity()
    layers["A"] = _make_act(activation, leakyrelu_hp=leakyrelu_hp)
    layers["D"] = nn.Dropout2d(p=dropout_2D) if (dropout_2D and dropout_2D > 0.0) else nn.Identity()

    seq = [layers[k] for k in order if k in layers]
    return nn.Sequential(*seq)


def _make_act(name: Optional[str], leakyrelu_hp) -> nn.Module:
    if not name or name.lower() in {"false", "none", "linear"}:
        return nn.Identity()
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=leakyrelu_hp, inplace=False)
    if name == "silu":
        return nn.SiLU(inplace=False)
    raise ValueError(f"Unknown activation: {name}")


def _make_norm(
    kind: Literal["bn", "gn", "in", "none"],
    num_channels: int,
    gn_groups: int = 16,
    bn_eps: float = 1e-3,
    bn_momentum: float = 0.1,
) -> nn.Module:
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


class _HybridPad2d(nn.Module):
    """
    Symmetric hybrid padding:
      - periodic in x
      - reflect in y
    """
    def __init__(self, pad_x: int, pad_y: int):
        super().__init__()
        self.pad_x = (pad_x, pad_x)
        self.pad_y = (pad_y, pad_y)

    def forward(self, x):
        x = F.pad(x, (self.pad_x[0], self.pad_x[1], 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.pad_y[0], self.pad_y[1]), mode="reflect")
        return x


def hybrid_pad_to_multiple(x: torch.Tensor, multiple: int = 16):
    """
    Pad H,W up to nearest multiple of `multiple`.
    Padding style:
      - x direction (W): circular
      - y direction (H): reflect

    Returns:
      x_pad, pad_info
    where pad_info = (pad_left, pad_right, pad_top, pad_bottom)
    """
    _, _, H, W = x.shape
    H_new = ((H + multiple - 1) // multiple) * multiple
    W_new = ((W + multiple - 1) // multiple) * multiple

    pad_h = H_new - H
    pad_w = W_new - W

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_w > 0:
        x = F.pad(x, (pad_left, pad_right, 0, 0), mode="circular")
    if pad_h > 0:
        x = F.pad(x, (0, 0, pad_top, pad_bottom), mode="reflect")

    return x, (pad_left, pad_right, pad_top, pad_bottom)


def hybrid_unpad(x: torch.Tensor, pad_info):
    pad_left, pad_right, pad_top, pad_bottom = pad_info
    H_end = x.shape[-2] - pad_bottom if pad_bottom > 0 else x.shape[-2]
    W_end = x.shape[-1] - pad_right if pad_right > 0 else x.shape[-1]
    return x[..., pad_top:H_end, pad_left:W_end]


class DoubleConvPhys(nn.Module):
    """
    Two conv_blocks back-to-back.
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
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvPhys(in_ch, out_ch, **kwargs)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpPixelShuffle(nn.Module):
    """
    Upsample by 2 using PixelShuffle, concat skip, then DoubleConvPhys.
    """
    def __init__(self, in_ch_from_down: int, in_ch_from_skip: int, out_ch: int, **kwargs):
        super().__init__()
        r = 2

        self.up = nn.Sequential(
            _HybridPad2d(pad_x=1, pad_y=1),
            nn.Conv2d(in_ch_from_down, out_ch * (r * r), kernel_size=3, padding=0, bias=True),
            nn.PixelShuffle(r),
        )

        self.conv = DoubleConvPhys(out_ch + in_ch_from_skip, out_ch, **kwargs)

    def forward(self, x_down: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_down = self.up(x_down)

        if x_down.shape[-2:] != x_skip.shape[-2:]:
            x_down = F.interpolate(
                x_down, size=x_skip.shape[-2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x_skip, x_down], dim=1)
        return self.conv(x)


class UNetPhysDepth1(nn.Module):
    """
    4-down / 4-up U-Net with optional ConvLSTM at the bottleneck.

    Accepts:
      - x: (B, C, H, W)
      - x: (B, T, C, H, W)

    For 56x56 input, automatically pads to 64x64 before encoding,
    then crops back to original H,W after decoding.
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
        )  # H, W

        self.down1 = Down(
            base, base * 2,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
            dropout_2D=dropout_2D,
        )  # H/2

        self.down2 = Down(
            base * 2, base * 4,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
            dropout_2D=dropout_2D,
        )  # H/4

        self.down3 = Down(
            base * 4, base * 8,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
            dropout_2D=dropout_2D,
        )  # H/8

        self.down4 = Down(
            base * 8, base * 16,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
            dropout_2D=dropout_2D,
        )  # H/16

        self.use_lstm = use_lstm
        if use_lstm:
            self.mid_lstm = BottleneckConvLSTM(
                in_channels=base * 16,
                hidden_channels=base * 16,
                kernel_size=3,
            )

        self.lstm_dropout = (
            nn.Dropout2d(p=lstm_dropout)
            if (use_lstm and lstm_dropout > 0)
            else nn.Identity()
        )

        # Decoder
        self.up4 = UpPixelShuffle(
            in_ch_from_down=base * 16,
            in_ch_from_skip=base * 8,
            out_ch=base * 8,
            kernel_size=kernel_size,
            activation=activation,
            batch_norm=batch_norm,
            dropout_2D=dropout_2D,
        )

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

    def _forward_single(self, x4d: torch.Tensor) -> torch.Tensor:
        """
        x4d: (B, C, H, W)
        """
        x4d, pad_info = hybrid_pad_to_multiple(x4d, multiple=16)

        x1 = self.enc1(x4d)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        y = self.head(x)

        y = hybrid_unpad(y, pad_info)
        return y

    def forward(self, x):
        # Case 1: single frame
        if x.dim() == 4:
            if self.use_lstm:
                # If you want single-frame inference with same model,
                # just bypass LSTM and use the encoder/decoder directly.
                return self._forward_single(x)
            else:
                return self._forward_single(x)

        # Case 2: sequence
        if x.dim() != 5:
            raise ValueError(f"Expected x dim 4 or 5, got {x.dim()} with shape {tuple(x.shape)}")

        B, T, C, H, W = x.shape

        # pad each frame consistently
        x_reshaped = x.reshape(B * T, C, H, W)
        x_reshaped, pad_info = hybrid_pad_to_multiple(x_reshaped, multiple=16)
        Hp, Wp = x_reshaped.shape[-2:]
        x = x_reshaped.reshape(B, T, C, Hp, Wp)

        x5_list = []
        x1_last = x2_last = x3_last = x4_last_skip = None

        for t in range(T):
            xt = x[:, t]
            x1 = self.enc1(xt)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

            x5_list.append(x5)
            x1_last, x2_last, x3_last, x4_last_skip = x1, x2, x3, x4

        x5_seq = torch.stack(x5_list, dim=1)  # (B, T, 16B, Hb, Wb)

        if self.use_lstm:
            x5_last = self.mid_lstm(x5_seq)
            x5_last = self.lstm_dropout(x5_last)
        else:
            x5_last = x5_seq[:, -1]

        x = self.up4(x5_last, x4_last_skip)
        x = self.up3(x, x3_last)
        x = self.up2(x, x2_last)
        x = self.up1(x, x1_last)
        y = self.head(x)

        y = hybrid_unpad(y, pad_info)
        return y