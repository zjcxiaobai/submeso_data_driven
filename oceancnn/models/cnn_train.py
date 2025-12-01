# load the following user-defined functions first
import importlib
import cnn_fun  
import data_norm
import cnn_hp
cnn_fun = importlib.reload(cnn_fun)  # reload the module
data_norm = importlib.reload(data_norm)
cnn_hp = importlib.reload(cnn_hp)
from cnn_fun import conv_block, Residual, reentrant_channel
from data_norm import MyDataset, prepare_data
from cnn_hp import _build_optimizer, _build_lr_scheduler, add_weight_decay
from cnn_device import _seed_everything, -set_num_threads, _save_checkpoints, _load_checkpoints
# other modules
import random
import torch
import torch.nn as nn
import os
import numpy as np
import pathlib 
import torch.optim as optim
import time
from torchmetrics.regression import MeanSquaredError, R2Score
from typing import Optional, Literal, Dict, Tuple
    
def run_trial(
    # trial parameters
    trial_id: int = 0, 
    data_path: str = "/scratch/jczhang/CNN_20251111/CNN_2/var_10169/", 
    checkpoint_dir: str = "./check_points",
    # CNN model parameter 
    conv_layers: int = 10, 
    kernel_size: int = 3, 
    lr: float = 0.00747, 
    batch_size: int = 8, 
    epochs: int = 200, 
    patience: int = 10, 
    activation: str = "SiLU",
    grad_clip: float | None = 1.0,
    use_channels_last: bool = False,
    batch_norm: str = "none",
    use_residule: bool = False,
    # filter
    filter_num: int = 8,  
    filter_mode:  Literal["const","linear_up"] = "const",
    max_filter: int = 256,
    # dropout - set 0 to turn off
    dropout_2D: float = 0.0,
    dropout_mode: str = "const",
    # optimizer
    weight_decay: float = 0.0,  # set 0 to turn off weight decay
    optim: Literal["Adam", "AdamW", "SGD", "NAG"] = "Adam",
    # learning rate schedule
    use_lr_schedule: bool = False,
    scheduler_monitor: str = "train_r2",
    
):
    # ------------------------ GPU or CPU ------------------------------
    # check how many CPUs available
    print("Number of CPUs:", os.cpu_count())
    _set_num_threads()
    # set the device, GPU or CPU
    device = _select_device()
    print(">>> device:", device)
    print(">>> torch.cuda.is_available():", torch.cuda.is_available())
    # ------------------------ save the parameters used in the model -----------------------------------
    out_path  = f"model_params_{trial_id}.pt"
    params = {
        "trial_id": int(trial_id),
        "conv_layers": int(conv_layers),
        "kernel_size": int(kernel_size),
        "filter_num": int(filter_num),
        "filter_mode": filter_mode,
        "max_filter": int(max_filter),
        "activation": activation,
        "lr": float(lr),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "patience": int(patience),
        "dropout_2D": float(dropout_2D),
        "dropout_mode": dropout_mode,
        "weight_decay": float(weight_decay),
        "grad_clip": grad_clip,
        "optimizer": optim,
        "batch_normalization": batch_norm,
    }
    torch.save({"params": params}, out_path)
    print(f"\n[Trial {trial_id}] config: layers={conv_layers}, kernel={kernel_size}, filter={filter_num}, "
          f"lr={lr:.5f}, batch={batch_size}, epochs={epochs}, patience={patience}, dropout={dropout_2D}, "
          f"adamW_lr={weight_decay}, activation={activation}")
    
    # ----------------------------------- load Data ------------------------------------------------
    X = np.load(os.path.join(data_path, "X.npy"))
    y = np.load(os.path.join(data_path, "y.npy"))
    train_loader, val_loader, test_loader, norm = prepare_data(X, y, device = device, batch_size=batch_size, num_workers=0, 
                                                         seed=1337, frac_train=0.6, frac_skip=0.15,
                                                        frac_val=0.15, frac_test=0.1, shuffle=False )
    # ---------------------------------- cnn block --------------------------------------------------
    model = reentrant_channel(
        input_channels=X.shape[1],
        conv_layers=conv_layers,
        filter_num = filter_num,
        filter_mode = filter_mode,
        max_filter = max_filter,
        kernel_size=kernel_size,
        activation = activation,
        batch_norm = batch_norm,
        head_channel = 1,
        use_residual = False,
        dropout_2D = dropout_2D,
        dropout_mode = dropout_mode,
    ).to(device)
    print(model) 
    print(">>> model param device:", next(model.parameters()).device)
    # --------------------------------------- optimizer and loss function ---------------------------------------
    optimizer = _build_optimizer(
        model,
        optim_name=optim,
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9,          
        betas=(0.9, 0.999),    # for Adam / AdamW
    )
    print("Using optimizer:", optimizer.__class__.__name__)
    print("Param groups:", len(optimizer.param_groups))
    for i, g in enumerate(optimizer.param_groups):
        print(f"  group {i}: lr={g['lr']}, weight_decay={g.get('weight_decay', 0.0)}")
    loss_fn = nn.MSELoss(reduction="mean")
    # build LR scheduler
    scheduler = _build_lr_scheduler(
        optimizer,
        use_scheduler = use_lr_schedule,
        monitor=scheduler_monitor,
        factor=0.5,
        patience=100,
        min_lr=1e-6,
    )
    # -------------------------------------- Resume check point if available ------------------------------------
    # create the checkpoints folder
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # create the file path of corresponding files
    latest_ckpt_path = os.path.join(checkpoint_dir, f"latest_trial_{trial_id}.pth")
    best_ckpt_path   = os.path.join(checkpoint_dir, f"best_val_trial_{trial_id}.pth")
    # initialize
    start_epoch = 0
    best_train_mse = float("inf")
    # resume if latest_ckpt_path exists
    if os.path.exists(latest_ckpt_path):   
        print(f"Resuming from checkpoint: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_train_mse = ckpt.get("best_train_mse", best_train_mse)
        start_epoch = ckpt.get("epoch", -1) + 1
        
    # ---- initialize best-by-test-R2 for training ----
    best_train = {
        "best_epoch": -1,
        "train_mse": float("inf"),
        "train_r2": float("-inf"),
    }
    # ============================== TRAINING LOOP ======================================
    for epoch_idx in range(start_epoch, epochs):
        t0 = time.time() # count time
        model.train()  # switch to training mode (e.g. active the BN and Dropout)
        train_mse = MeanSquaredError().to(device)
        train_r2  = R2Score().to(device)
        
        running_loss, n_seen = 0.0, 0  # sum of all loss over all samples, count of samples processed so far
        #  ---- training ----
        for data, target in train_loader:     # iterate over the mini batches
            if use_channels_last and data.ndim == 4 and device.type == "cuda":
                data = data.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            else:
                data = data.to(device, non_blocking=True)   # move the data to the chosen device
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            output = model(data) # compute the prediction
            if output.ndim == 4 and target.ndim == 3:
                target = target.unsqueeze(1)
            loss = loss_fn(output, target)  # mean squared error
            loss.backward() # compute the gradient with respect to the model parameters
            if grad_clip is not None:  # help stabilize training
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()  # apply adam updates using the accumulated grads

            bs = data.size(0)   # batch size
            running_loss += loss.item() * bs  # for sample weighted mean of MSE
            n_seen += bs  # count samples
            train_mse.update(output.view(-1), target.view(-1))
            train_r2.update(output.view(-1), target.view(-1))

        avg_train_loss = running_loss / max(1, n_seen)  # compute the average training loss for the epoch

        tr_mse = float(train_mse.compute().item())
        tr_r2  = float(train_r2.compute().item())
        dt = time.time() - t0

        # apply learning rate schedule
        if scheduler is not None:
            if scheduler_monitor == "train_r2":
                scheduler.step(tr_r2)
            else:  # "train_mse"
                scheduler.step(tr_mse)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch_idx+1:4d}/{epochs} | "
              f"trainLoss {avg_train_loss:.6f} | trainMSE {tr_mse:.6f} | "
              f"trainR2 {tr_r2:+.4f} | lr={current_lr:.2e} | {dt:.1f}s")
        # ------------------------------ early stopping ----------------------------------------
        improved = tr_mse < best_train_mse - 1e-12
        if improved:
            best_train_mse = tr_mse
            best_train.update({
                "best_epoch": epoch_idx + 1,
                "train_r2": float(tr_r2),
                "train_mse": float(tr_mse),
            })
            torch.save({
                "epoch": epoch_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_train_mse": best_train_mse,
            }, best_ckpt_path)

        # always save "latest" to resume
        torch.save({
            "epoch": epoch_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_train_mse": best_train_mse,
        }, latest_ckpt_path)

    # --------------------------- FINAL TEST EVAL ------------------------
    # Load the best-by-val checkpoint
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    final_mse = MeanSquaredError().to(device)
    final_r2  = R2Score().to(device)
    with torch.no_grad():
        for data, target in train_loader:
            data   = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            out    = model(data)
            if out.ndim == 4 and target.ndim == 3:
                target = target.unsqueeze(1)
            final_mse.update(out.view(-1), target.view(-1))
            final_r2.update(out.view(-1), target.view(-1))
    tr_mse_final = float(final_mse.compute().item())
    tr_r2_final  = float(final_r2.compute().item())
    # ------------------ save the final test score ------------------------
    epochs_used = epochs
    bundle = {
        "params": params,
        "best_by_train": best_train,
        "epochs_used": int(epochs_used),
        "final_train": {"train_mse": tr_mse_final, "train_r2": tr_r2_final}
    }
    torch.save(bundle, out_path)
    # print final test score
    print(f"[Trial {trial_id}] BEST (by TRAIN) at epoch {best_train['best_epoch']}: "
          f"trainR2={best_train['train_r2']:+.4f}, trainMSE={best_train['train_mse']:.6f} | "
          f"FINAL TRAIN → R2={tr_r2_final:+.4f}, MSE={tr_mse_final:.6f} | epochs_used={epochs_used}")
    
if __name__ == "__main__":
    # path
    data_path = "/scratch/jczhang/CNN_20251111/CNN_2/var_10169/"
    ckpt_root = "./check_points"
    # parameters
    conv_layers = 12
    kernel_size = 3
    filter_num = 10
    lr = 0.00747
    batch_size  = 8
    patience    = 300000
    epochs_fixed = 5000
    activation = "SiLU"
    dropout_2D = 0
    adamW_lr = 3e-4

    idx = 95
    run_trial(
        # trial parameters
        trial_id= idx,
        data_path=data_path,
        checkpoint_dir=ckpt_root,
        # CNN model parameters
        conv_layers=conv_layers,
        kernel_size=kernel_size,
        lr = lr,
        batch_size=batch_size,
        epochs=int(epochs_fixed),
        patience=int(patience),
        activation = activation,
        grad_clip = None,
        use_channels_last = False,
        batch_norm = "none",
        use_residual = False,
        # filter
        filter_num = filter_num, 
        filter_mode = "const",
        max_filter = 12,
        # dropout
        dropout_2D = 0.0,
        dropout_mode = "const",
        # optimizer
        weight_decay = 0.0,
        optim = "Adam",
        # learning rate schedule
        use_lr_schedule = False,
        schedule_monitor = "train_r2",
    )

