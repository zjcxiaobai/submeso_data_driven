# transformer_spatiotemporal_train.py

import os
import time
import math
import pathlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.regression import MeanSquaredError, R2Score

# Your utilities
import cnn_device
from cnn_device import _seed_everything, _select_device
import data_loader_window


# ------------------------- model -------------------------

class PatchEmbed(nn.Module):
    """
    (B,C,H,W) -> (B, N, D) where N = (H/P)*(W/P)
    Uses conv with stride=patch to create patch tokens.
    """
    def __init__(self, in_ch: int, dim: int, patch: int):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch, padding=0)

    def forward(self, x):
        # x: (B,C,H,W)
        x = self.proj(x)                 # (B,D,Hp,Wp)
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, N, D), N=Hp*Wp
        return x, (Hp, Wp)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""
    def __init__(self, dim: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, L, D)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class SpatioTemporalPatchTransformer(nn.Module):
    """
    Input : (B,T,C,H,W) or (B,C,H,W)
    Output: (B,1,H,W)  (predict map for LAST time step)

    Tokens are patches across space and time:
      total tokens = T * (H/P)*(W/P)
    """
    def __init__(
        self,
        in_ch: int = 5,
        out_ch: int = 1,
        H: int = 88,
        W: int = 56,
        T_max: int = 64,          # max supported window length for time embedding
        patch: int = 8,           # 88x56 -> 11x7 patches when patch=8
        dim: int = 256,
        depth: int = 6,
        nhead: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert H % patch == 0 and W % patch == 0, "Choose patch that divides H and W exactly."
        self.H, self.W = H, W
        self.patch = patch
        self.dim = dim

        self.patch_embed = PatchEmbed(in_ch, dim, patch)
        Hp, Wp = H // patch, W // patch
        self.Hp, self.Wp = Hp, Wp
        self.Np = Hp * Wp

        # Positional embeddings
        self.pos_spatial = nn.Parameter(torch.zeros(1, self.Np, dim))
        self.pos_time    = nn.Parameter(torch.zeros(1, T_max, dim))

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, nhead, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # Decode last-frame tokens back to map
        # tokens (B, Np, D) -> (B, D, Hp, Wp) -> upsample to (B, out_ch, H, W)
        self.decode = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, out_ch, kernel_size=1, bias=True),
        )

    def forward(self, x):
        # Accept (B,C,H,W) or (B,T,C,H,W)
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B,1,C,H,W)

        if x.dim() != 5:
            raise ValueError(f"Expected 4D or 5D input, got {x.shape}")

        B, T, C, H, W = x.shape
        assert H == self.H and W == self.W, f"Model expects H,W=({self.H},{self.W}), got ({H},{W})"
        if T > self.pos_time.shape[1]:
            raise ValueError(f"T={T} exceeds T_max={self.pos_time.shape[1]}")

        # Build tokens for all frames
        tokens_all = []
        for t in range(T):
            xt = x[:, t]                      # (B,C,H,W)
            tok, _ = self.patch_embed(xt)     # (B,Np,D)
            tok = tok + self.pos_spatial      # spatial pos
            tok = tok + self.pos_time[:, t:t+1, :]  # time pos broadcast over Np? (B,1,D)
            tok = tok + 0.0  # keep as explicit tensor op
            # Broadcast time pos across Np tokens:
            tok = tok + self.pos_time[:, t:t+1, :].expand(B, self.Np, self.dim)
            tokens_all.append(tok)

        z = torch.cat(tokens_all, dim=1)  # (B, T*Np, D)

        # Transformer encoder
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)  # (B, T*Np, D)

        # Take last frame tokens only
        z_last = z[:, (T-1)*self.Np : T*self.Np, :]     # (B, Np, D)
        z_last = z_last.transpose(1, 2).contiguous()    # (B, D, Np)
        z_last = z_last.view(B, self.dim, self.Hp, self.Wp)  # (B, D, Hp, Wp)

        # Upsample back to full resolution
        y = F.interpolate(z_last, size=(self.H, self.W), mode="bilinear", align_corners=False)
        y = self.decode(y)  # (B, out_ch, H, W)
        return y


# ------------------------- eval helpers -------------------------

@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    mse = MeanSquaredError().to(device)
    r2  = R2Score().to(device)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if yb.ndim == 3:
            yb = yb.unsqueeze(1)
        pred = model(xb)
        mse.update(pred.view(-1), yb.view(-1))
        r2.update(pred.view(-1), yb.view(-1))

    return float(mse.compute().item()), float(r2.compute().item())


# ------------------------- training -------------------------

def run_transformer_trial(
    trial_id: int = 0,
    data_path: str = "/scratch/jczhang/data_5km/save_coarse/out_npz/",
    checkpoint_dir: str = "./check_points_transformer",

    # window length
    T: int = 14,

    # model
    patch: int = 8,
    dim: int = 256,
    depth: int = 6,
    nhead: int = 8,
    dropout: float = 0.0,

    # training
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    batch_size: int = 64,
    epochs: int = 200,
    patience: int = 10,
    grad_clip: float | None = 1.0,

    # reproducibility
    seed: int = 42,
    deterministic: bool = False,
):
    _seed_everything(seed=seed, deterministic=deterministic)
    device = _select_device()
    print(">>> device:", device)
    print(">>> patience:", patience)

    X = np.load(os.path.join(data_path, "X.npy"))  # (N,C,H,W)
    y = np.load(os.path.join(data_path, "y.npy"))  # (N,H,W) or (N,1,H,W)
    print(">>> X:", X.shape, "y:", y.shape)

    # Windowed loader: xb should be (B,T,C,H,W)
    train_loader, val_loader, test_loader, norm = data_loader_window.prepare_data_windowed(
        X, y,
        device=device,
        T=T,
        batch_size=batch_size,
        num_workers=0,
        seed=seed,
        frac_train=0.8,
        frac_skip=0.0,
        frac_val=0.1,
        frac_test=0.1,
        shuffle=False,          # don't shuffle raw time axis
        normalize_x=True,
        normalize_y=True,
    )

    # Infer H,W,C from X
    C = int(X.shape[1])
    H = int(X.shape[2])
    W = int(X.shape[3])

    model = SpatioTemporalPatchTransformer(
        in_ch=C,
        out_ch=1,
        H=H,
        W=W,
        T_max=max(64, T),
        patch=patch,
        dim=dim,
        depth=depth,
        nhead=nhead,
        dropout=dropout,
    ).to(device)

    print(model)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, f"best_trial_{trial_id}.pth")

    best_val = float("inf")
    bad = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        tr_mse_metric = MeanSquaredError().to(device)
        tr_r2_metric  = R2Score().to(device)

        running = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if yb.ndim == 3:
                yb = yb.unsqueeze(1)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

            bs = xb.size(0)
            running += float(loss.item()) * bs
            seen += bs

            tr_mse_metric.update(pred.view(-1), yb.view(-1))
            tr_r2_metric.update(pred.view(-1), yb.view(-1))

        tr_loss = running / max(1, seen)
        tr_mse  = float(tr_mse_metric.compute().item())
        tr_r2   = float(tr_r2_metric.compute().item())

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
            torch.save({"model": model.state_dict(), "best_val_mse": best_val}, best_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping (no val improvement for {patience} epochs).")
                break

    # Load best
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(">>> loaded best:", best_path, "best_val_mse=", ckpt.get("best_val_mse", None))

    tr_mse, tr_r2 = eval_loader(model, train_loader, device)
    va_mse, va_r2 = eval_loader(model, val_loader, device)
    te_mse, te_r2 = eval_loader(model, test_loader, device)

    print("FINAL TRAIN →", f"R2={tr_r2:+.4f}", f"MSE={tr_mse:.6f}")
    print("FINAL VAL   →", f"R2={va_r2:+.4f}", f"MSE={va_mse:.6f}")
    print("FINAL TEST  →", f"R2={te_r2:+.4f}", f"MSE={te_mse:.6f}")

    return {"train": (tr_mse, tr_r2), "val": (va_mse, va_r2), "test": (te_mse, te_r2)}


if __name__ == "__main__":
    run_transformer_trial(
        trial_id=1,
        data_path="/scratch/jczhang/data_5km/save_coarse/out_npz/",
        checkpoint_dir="./check_points_transformer",
        T=14,

        patch=8,       # 88x56 -> 11x7 patches => tokens = 14*77=1078
        dim=256,
        depth=6,
        nhead=8,
        dropout=0.0,

        lr=5e-4,
        weight_decay=0.0,
        batch_size=64,
        epochs=200,
        patience=10,
        grad_clip=1.0,

        seed=42,
        deterministic=False,
    )
