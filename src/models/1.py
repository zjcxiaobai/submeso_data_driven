import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.regression import R2Score, MeanSquaredError

# user-defined function
from cnn_device import _seed_everything, _select_device
import data_loader

# ------------------------- shape helpers -------------------------

def crop_or_pad_to(x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """
    Make x match target H,W by:
      - cropping bottom/right if too large
      - padding bottom/right if too small
    x: (B,C,H,W)
    """
    th, tw = target_hw
    _, _, h, w = x.shape

    # crop
    x = x[:, :, :min(h, th), :min(w, tw)]
    _, _, h2, w2 = x.shape

    # pad
    pad_h = max(0, th - h2)
    pad_w = max(0, tw - w2)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x


# ------------------------- dataset + normalization -------------------------

class NpyFieldDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N,C,H,W)
        # y: (N,H,W) or (N,1,H,W)
        assert X.ndim == 4, f"Expected X (N,C,H,W), got {X.shape}"
        if y.ndim == 3:
            y = y[:, None, :, :]
        assert y.ndim == 4, f"Expected y (N,1,H,W), got {y.shape}"
        assert X.shape[0] == y.shape[0]
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


class Standardizer:
    """Channel-wise standardization for X; optional global standardization for y."""
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.x_mean = None
        self.x_std = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, norm_y: bool = False):
        # X: (N,C,H,W)
        self.x_mean = X.mean(axis=(0, 2, 3), keepdims=True)
        self.x_std  = X.std(axis=(0, 2, 3), keepdims=True) + self.eps
        if norm_y:
            if y.ndim == 3:
                y = y[:, None, :, :]
            self.y_mean = float(y.mean())
            self.y_std  = float(y.std() + self.eps)

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std + self.y_mean


# ------------------------- model blocks -------------------------

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, use_bn=True, act="silu"):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = nn.SiLU() if act.lower() == "silu" else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SpatialLSTMBlock(nn.Module):
    """
    Paper-style "spatial LSTM":
      x (B,C,Hb,Wb) -> seq (B,L,C) where L=Hb*Wb -> LSTM -> (B,C,Hb,Wb)
    """
    def __init__(self, channels: int, hidden_size: int = 1024, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.proj = nn.Linear(hidden_size, channels) if hidden_size != channels else nn.Identity()

    def forward(self, x):
        B, C, Hb, Wb = x.shape
        L = Hb * Wb

        seq = x.view(B, C, L).transpose(1, 2).contiguous()  # (B,L,C)
        out, _ = self.lstm(seq)                              # (B,L,H)
        out = self.proj(out)                                 # (B,L,C)
        y = out.transpose(1, 2).contiguous().view(B, C, Hb, Wb)
        return y


class PaperLikeUNetSpatialLSTM(nn.Module):
    """
    Same idea as your Keras code (U-Net + spatial LSTM at bottleneck),
    but robust to your input size (88x56) via dynamic crop/pad.
    Output is linear (y can be negative).
    """
    def __init__(self, in_ch: int, use_bn: bool = True, act: str = "silu"):
        super().__init__()

        # "same" pooling behavior like Keras padding='same'
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # Encoder
        self.c1a = ConvBNAct(in_ch, 128, k=3, use_bn=use_bn, act=act)
        self.c1b = ConvBNAct(128, 256, k=3, use_bn=use_bn, act=act)

        self.c2a = ConvBNAct(256, 256, k=3, use_bn=use_bn, act=act)
        self.c2b = ConvBNAct(256, 512, k=3, use_bn=use_bn, act=act)

        self.c3a = ConvBNAct(512, 512, k=3, use_bn=use_bn, act=act)
        self.c3b = ConvBNAct(512, 1024, k=3, use_bn=use_bn, act=act)

        self.c4a = ConvBNAct(1024, 1024, k=3, use_bn=use_bn, act=act)
        self.c4b = ConvBNAct(1024, 1024, k=3, use_bn=use_bn, act=act)

        # Spatial LSTM (paper uses 2 layers of 1024)
        self.spatial_lstm = SpatialLSTMBlock(channels=1024, hidden_size=1024, num_layers=2, dropout=0.0)

        # Decoder
        self.up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.c5a = ConvBNAct(512 + 1024 + 512, 256, k=3, use_bn=use_bn, act=act)
        self.c5b = ConvBNAct(256, 256, k=3, use_bn=use_bn, act=act)

        self.up6 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.c6a = ConvBNAct(256 + 512 + 256, 128, k=3, use_bn=use_bn, act=act)
        self.c6b = ConvBNAct(128, 128, k=3, use_bn=use_bn, act=act)

        self.up7 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.c7a = ConvBNAct(128 + 256, 64, k=3, use_bn=use_bn, act=act)
        self.c7b = ConvBNAct(64, 64, k=3, use_bn=use_bn, act=act)

        self.head = nn.Conv2d(64, 1, kernel_size=1)  # linear output

    def forward(self, x):
        # x: (B,C,H,W)

        # enc1
        c1 = self.c1b(self.c1a(x))      # (B,256,H,W)
        p1 = self.pool(c1)

        # enc2
        c2 = self.c2b(self.c2a(p1))     # (B,512,H2,W2)
        p2 = self.pool(c2)

        # enc3
        c3 = self.c3b(self.c3a(p2))     # (B,1024,H3,W3)
        p3 = self.pool(c3)

        # bottleneck
        c4 = self.c4b(self.c4a(p3))     # (B,1024,Hb,Wb)
        c4 = self.spatial_lstm(c4)

        # dec5: up + concat(u5, c3, c2 cropped to c3)
        u5 = self.up5(c4)
        u5 = crop_or_pad_to(u5, c3.shape[-2:])
        c2c = crop_or_pad_to(c2, c3.shape[-2:])
        x5 = torch.cat([u5, c3, c2c], dim=1)
        c5 = self.c5b(self.c5a(x5))     # (B,256, H3, W3)

        # dec6: up + concat(u6, c2, c1 cropped to c2)
        u6 = self.up6(c5)
        u6 = crop_or_pad_to(u6, c2.shape[-2:])
        c1c = crop_or_pad_to(c1, c2.shape[-2:])
        x6 = torch.cat([u6, c2, c1c], dim=1)
        c6 = self.c6b(self.c6a(x6))     # (B,128, H2, W2)

        # dec7: up + concat(u7, c1)
        u7 = self.up7(c6)
        u7 = crop_or_pad_to(u7, c1.shape[-2:])
        x7 = torch.cat([u7, c1], dim=1)
        c7 = self.c7b(self.c7a(x7))     # (B,64,H,W)

        y = self.head(c7)               # (B,1,H,W)
        return y


# ------------------------- train/eval -------------------------

@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    mse = MeanSquaredError().to(device)
    r2 = R2Score().to(device)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb)
        mse.update(pred.view(-1), yb.view(-1))
        r2.update(pred.view(-1), yb.view(-1))
    return float(mse.compute().item()), float(r2.compute().item())


def run(
    data_path: str,
    batch_size: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    epochs: int = 500,
    patience: int = 10,
    seed: int = 42,
    deterministic: bool = False,
    use_bn: bool = True,
    norm_y: bool = False,
):
    _seed_everything(seed, deterministic)
    device = _select_device()
    print(">>> device:", device)
    print(">>> patience:", patience)

    X = np.load(os.path.join(data_path, "X.npy"))  # (N,C,H,W)
    y = np.load(os.path.join(data_path, "y.npy"))  # (N,H,W) or (N,1,H,W)
    print(X.shape)
    train_loader, val_loader, test_loader, norm = data_loader.prepare_data(
        X, y,
        device=device,
        batch_size=batch_size,
        num_workers=0,   # see RNG note below
        seed=seed,
        frac_train=0.8,
        frac_skip=0.0,
        frac_val=0.1,
        frac_test=0.1,
        shuffle=False,   # DO NOT shuffle the raw time axis
        normalize_x=True,
        normalize_y=True,
    )
    # ds_full = NpyFieldDataset(X, y)
    # N = len(ds_full)
    # n_train = int(0.8 * N)
    # n_val = int(0.1 * N)
    # n_test = N - n_train - n_val

    # g = torch.Generator().manual_seed(seed)
    # train_ds, val_ds, test_ds = random_split(ds_full, [n_train, n_val, n_test], generator=g)

    # # fit normalizer on train only
    # stdz = Standardizer()
    # stdz.fit(X[train_ds.indices], y[train_ds.indices], norm_y=norm_y)

    # def make_split(split):
    #     idx = split.indices
    #     Xs = stdz.transform_X(X[idx])
    #     ys = y[idx]
    #     if ys.ndim == 3:
    #         ys = ys[:, None, :, :]
    #     ys = stdz.transform_y(ys.astype(np.float32)) if norm_y else ys.astype(np.float32)
    #     return NpyFieldDataset(Xs, ys)

    # train_ds = make_split(train_ds)
    # val_ds = make_split(val_ds)
    # test_ds = make_split(test_ds)

    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = PaperLikeUNetSpatialLSTM(in_ch=X.shape[1], use_bn=use_bn, act="silu").to(device)
    print(model)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    bad = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        tr_mse_metric = MeanSquaredError().to(device)
        tr_r2_metric = R2Score().to(device)

        running = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            bs = xb.size(0)
            running += float(loss.item()) * bs
            seen += bs

            tr_mse_metric.update(pred.view(-1), yb.view(-1))
            tr_r2_metric.update(pred.view(-1), yb.view(-1))

        tr_loss = running / max(1, seen)
        tr_mse = float(tr_mse_metric.compute().item())
        tr_r2 = float(tr_r2_metric.compute().item())

        va_mse, va_r2 = eval_loader(model, val_loader, device)
        dt = time.time() - t0

        print(
            f"Epoch {epoch:4d}/{epochs} | "
            f"trainLoss {tr_loss:.6f} trainMSE {tr_mse:.6f} trainR2 {tr_r2:+.4f} || "
            f"valMSE {va_mse:.6f} valR2 {va_r2:+.4f} | "
            f"bad_epochs {bad:3d}/{patience} | {dt:.1f}s"
        )

        if va_mse < best_val - 1e-12:
            best_val = va_mse
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping (no val improvement for {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    tr_mse, tr_r2 = eval_loader(model, train_loader, device)
    va_mse, va_r2 = eval_loader(model, val_loader, device)
    te_mse, te_r2 = eval_loader(model, test_loader, device)

    print("FINAL TRAIN →", f"R2={tr_r2:+.4f}", f"MSE={tr_mse:.6f}")
    print("FINAL VAL   →", f"R2={va_r2:+.4f}", f"MSE={va_mse:.6f}")
    print("FINAL TEST  →", f"R2={te_r2:+.4f}", f"MSE={te_mse:.6f}")

    return model


# ------------------------- main -------------------------

if __name__ == "__main__":
    run(
        data_path="/scratch/jczhang/data_5km/save_coarse/out_npz/",
        
        # data_path = "/scratch/jczhang/data_5km/var_6000/",
        batch_size=64,
        lr=5e-4,
        weight_decay=0.0,
        epochs=500,
        patience=10,
        seed=42,
        deterministic=False,
        use_bn=True,
        norm_y=False,
    )