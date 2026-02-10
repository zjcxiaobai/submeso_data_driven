#  pytorch
import torch
import torch.nn as nn
# General
import numpy as np
from typing import Optional, Literal, Dict


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
    if kind == "none": return nn.Identity()
    if kind == "bn":   return nn.BatchNorm2d(num_channels, eps=bn_eps, momentum=bn_momentum, affine=True)
    if kind == "in":   return nn.InstanceNorm2d(num_channels, eps=1e-5, affine=True, track_running_stats=False)
    if kind == "gn":   return nn.GroupNorm(num_groups=min(gn_groups, num_channels), num_channels=num_channels, eps=1e-5)
    raise ValueError(f"Unknown norm kind: {kind}")

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
    
#  filter number mode: constant or linear
def _make_filter(conv_layers: int, mode: str, base: int = 64, last_cap: int = 96):
    L = max(0, conv_layers - 1)
    if mode.lower() == "const":
        return [base] * L
    if mode.lower() == "linear_up":
        if L == 0:
            return []
        if L == 1:
            return [last_cap]
        return [int(round(base + (last_cap - base) * i / (L - 1))) for i in range(L)]
    if mode.lower() == "linear_down_3":
        if L == 0:
            return []
        if L == 1:
            # Only one hidden: just use last_cap
            return [last_cap]
        if L == 2:
            # Two hiddens: go directly from last_cap -> base
            return [last_cap, base]
        # L >= 3: first 3 hidden layers form a linear ramp last_cap -> base
        n_transition = 3  # number of layers in the ramp
        vals = [
            int(round(last_cap + (base - last_cap) * i / (n_transition - 1)))
            for i in range(n_transition)
        ]
        # Remaining layers stay at base
        if L > n_transition:
            vals.extend([base] * (L - n_transition))
        return vals

    if mode.lower() == "three_step":
        # First 3: 16, next 3: 8, rest: 4
        if L == 0:
            return []
        filters = []
        for i in range(L):
            if i < 3:
                filters.append(16)
            elif i < 6:
                filters.append(8)
            else:
                filters.append(4)
        return filters
        
    raise TypeError("filter_num must be int | str | list[int].")
    
# residual connection - robust 
class Residual(nn.Module):
    def __init__(self, main: nn.Module, in_ch: int, out_ch: int,
                 use_norm_on_proj: Literal["bn","gn","in","none"]="none"):
        super().__init__()
        self.main = main
        if in_ch != out_ch:
            proj_layers = [nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)]
            # optional: keep scales consistent if you use norms elsewhere
            if use_norm_on_proj == "bn":
                proj_layers.append(nn.BatchNorm2d(out_ch))
            elif use_norm_on_proj == "in":
                proj_layers.append(nn.InstanceNorm2d(out_ch, affine=True, track_running_stats=False))
            elif use_norm_on_proj == "gn":
                g = max(1, min(16, out_ch))
                while out_ch % g != 0: g -= 1
                proj_layers.append(nn.GroupNorm(g, out_ch))
            self.proj = nn.Sequential(*proj_layers)

            # NOTE: if you really want the residual path to dominate at init,
            # zero the LAST norm/scale in `self.main` instead of zeroing `proj`.
            # If you keep the next line, you disable the skip at init.
            # nn.init.zeros_(self.proj[0].weight)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.main(x) + self.proj(x)

# dropout function
def _make_dropout(
    i: int,
    mode: str = "const",
    p_max: float = 0.0,
    num_hidden: int = 10,
) -> float:
    """Dropout for hidden block i (0..num_hidden-1)."""
    if mode.lower() == "const":
        return float(p_max)

    if mode.lower() == "linear_up":
        # i = 0           -> frac = 0.0
        # i = num_hidden-1 -> frac = 1.0
        frac = i / (num_hidden - 1)
        return float(p_max * frac)
        
    return 0.0

# CNN
class reentrant_channel(nn.Module):       
    def __init__(
        self,
        input_channels: int = 8,
        conv_layers: int = 8,
        filter_num: int = 4,
        filter_mode: Literal["const","linear_up"] = "const",
        max_filter: int = 12,
        kernel_size: int = 3, 
        activation: str = 'SiLU',
        batch_norm: Literal["bn","gn","in","none"] = "bn",
        head_channel: int = 1,
        use_residual: bool = False,
        dropout_2D: float = 0.0, 
        dropout_mode: str = "const",
    ):
        super(reentrant_channel, self).__init__()
        assert conv_layers >= 1, "conv_layers must be >= 1"
        
        # number of hidden layers
        num_hidden = conv_layers - 1
            
        # filters
        filters = _make_filter(
            conv_layers = conv_layers,
            mode = filter_mode,        
            base = filter_num,          
            last_cap = max_filter  
        )
            
        # initialize the layers
        layers: list[nn.Module] = []
        # building the model
        in_channels = input_channels
        for i in range(num_hidden):
            out_channels = filters[i] if i < len(filters) else filters[-1]
            p = _make_dropout(i, num_hidden=num_hidden, p_max=dropout_2D, mode= dropout_mode)    # dropout      
            # --- kernel: first block 5x5, the rest 3x3 ---
            ks = 5 if i == 0 else kernel_size
            # layers settings
            layer = conv_block(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=ks,
                activation=activation,
                batch_norm=batch_norm,
                order="CNAD",
                dropout_2D=p,
            )
            #  use residual
            if use_residual:
                layers.append(Residual(layer, in_channels, out_channels,
                               use_norm_on_proj="none"))
            else:
                layers.append(layer)
            in_channels = out_channels
            
        # last layer
        head = conv_block(
            in_channels=in_channels,
            out_channels= head_channel,
            kernel_size=1,
            activation="False",   # no activation
            batch_norm="none",
            order="C",            # no batch normalization
            dropout_2D = 0.0
        )
        layers.append(head)
        self.net = nn.Sequential(*layers) # convert to a sequential model
        self._initialize_weights(activation=activation)
        
    def forward(self, x):
        return self.net(x)

    def _initialize_weights(self, activation: str = "ReLU"):
        # Map to a valid kaiming nonlinearity token
        nl = {
            "ReLU": "relu",
            "LeakyReLU": "leaky_relu",
            "SiLU": "relu",   
            "Swish": "relu",
            "False": "linear",
            "Linear": "linear",
            "None": "linear",
        }.get(activation, "relu")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity=nl)  # fan_out for relu, fan_in for silu
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
