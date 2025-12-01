



# seeds for the random processes
def _seed_everything(seed: int = 1337):
    random.seed(seed)  # set seed for random module          
    np.random.seed(seed) # set seed for Numpy
    torch.manual_seed(seed) # set seed for pytorch
# select devices
def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
    
# set the number of cpus
def _set_num_threads():
    env_cpus = os.environ.get("SLURM_CPUS_PER_TASK")  # how many cpus I can use for this task
    n = int(env_cpus) if env_cpus else (os.cpu_count() or 1) # cpus I can use 
    torch.set_num_threads(max(1, n))  # set threads number to parallelize CPU 
    print(f"Using {torch.get_num_threads()} CPU threads")
    
# save check point
def _save_checkpoint(path, epoch_idx, model, optimizer, best_test_mse, rng=None):
    ckpt = {
        "epoch": epoch_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_test_mse": best_test_mse,
        "rng_state": rng or {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(ckpt, path)

# load checkpoint
def _load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    return ckpt