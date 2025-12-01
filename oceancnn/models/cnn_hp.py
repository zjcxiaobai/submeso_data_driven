

# optimizer options
def _build_optimizer(
    model: nn.Module,
    optim_name: Literal["Adam", "AdamW", "SGD", "NAG"] = "Adam",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    betas: Tuple[float, float] = (0.9, 0.999),
) -> torch.optim.Optimizer:
    """
    Build an optimizer with optional decoupled weight decay.

    optim_name:
        "Adam"  -> torch.optim.Adam
        "AdamW" -> torch.optim.AdamW
        "SGD"   -> torch.optim.SGD
        "NAG"   -> torch.optim.SGD with nesterov=True
    """
    name = optim_name.lower()

    # use your add_weight_decay helper if wd > 0
    if weight_decay > 0.0:
        params = add_weight_decay(model, weight_decay)
    else:
        params = model.parameters()

    if name == "adam":
        opt = optim.Adam(params, lr=lr, betas=betas)
    elif name == "adamw":
        opt = optim.AdamW(params, lr=lr, betas=betas)
    elif name == "sgd":
        opt = optim.SGD(params, lr=lr, momentum=momentum)
    elif name == "nag":
        opt = optim.SGD(params, lr=lr, momentum=momentum, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer '{optim_name}'. Use 'Adam', 'AdamW', 'SGD', or 'NAG'.")

    return opt

#  learning rate schedule options
def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    use_scheduler: bool,
    monitor: str = "train_r2",
    factor: float = 0.5,
    patience: int = 100,
    min_lr: float = 1e-6,
) -> Optional[torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """
    Optionally build a ReduceLROnPlateau scheduler.

    monitor: "train_r2" (maximize) or "train_mse" (minimize).
    If use_scheduler is False, returns None.
    """
    if not use_scheduler:
        return None

    if monitor == "train_r2":
        mode = "max"
    elif monitor == "train_mse":
        mode = "min"
    else:
        raise ValueError(f"Unknown monitor '{monitor}', expected 'train_r2' or 'train_mse'.")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=1e-4,
        min_lr=min_lr,
    )
    return scheduler

# apply weight decay only for some weights
def add_weight_decay(model, wd, skip=('bias','bn.weight','bn.bias',
                                     'BatchNorm.weight','BatchNorm.bias','LayerNorm.weight')):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (no_decay if any(s in n for s in skip) else decay).append(p)
    return [{'params': decay, 'weight_decay': wd},
            {'params': no_decay, 'weight_decay': 0.0}]
    