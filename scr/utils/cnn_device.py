import os
import random
import numpy as np
import torch


# make sure all the random process fix
def _seed_everything(seed: int = 42, deterministic: bool = False):
    # Python / NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Determinism (optional)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

# select a device that fit
def _select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  
def _set_num_threads():
    env_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    n = int(env_cpus) if env_cpus else (os.cpu_count() or 1)
    torch.set_num_threads(max(1, n))
    print(f"Using {torch.get_num_threads()} CPU threads")


def _save_checkpoint(path, epoch_idx, model, optimizer, best_val_mse, rng_state=None):
    ckpt = {
        "epoch": epoch_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_mse": best_val_mse,
        "rng_state": rng_state or {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(ckpt, path)

#  load the check point files
def _load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)

# same random parameters if restart from a check point
def _restore_rng_state(ckpt):
    rs = ckpt.get("rng_state", None)
    if rs is None:
        return
    random.setstate(rs["python"])
    np.random.set_state(rs["numpy"])
    torch.set_rng_state(rs["torch"])
    if torch.cuda.is_available() and rs.get("cuda", None) is not None:
        torch.cuda.set_rng_state_all(rs["cuda"])
