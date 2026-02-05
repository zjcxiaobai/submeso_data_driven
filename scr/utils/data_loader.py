#  pytorch
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
# General
import numpy as np

# ------------------------------- Z-score normalization -------------------------------------
def _fit_z_norm_X(X_train):
    """
    X_train: (N, C, H, W) or (N, C, ...) -> per-channel mu, sd from TRAIN.
    This function computes the z-score normalization of X
    Which transform the mean = 0, and standard deviation = 1
    mean and std computed per channel [1, C, 1, 1]
    """
    # mean/std over N and all spatial dims, per channel
    axes = tuple(i for i in range(X_train.ndim) if i not in (1,))  # normalize all dims except channel dim
    mu = X_train.mean(axis=axes, keepdims=True)                  # shape (1, C, 1, 1)
    sd = X_train.std(axis=axes, ddof=1, keepdims=True)    # shape (1, C, 1, 1)
    return mu.astype(np.float32), sd.astype(np.float32)

def _fit_z_norm_y(y_train):
    """
    y_train: (N, 1, H, W) or (N, H, W) or (N,)
    """
    mu = y_train.mean()
    sd = y_train.std(ddof=1)
    return np.float32(mu), np.float32(sd)

#  apply the az-score normalization to X_train, X_val, X_test
def _apply_z_norm_X(X, mu, sd):
    return (X - mu) / sd

def _apply_z_norm_y(y, mu, sd):
    return (y - mu) / sd

#  ------------------------------ custom pytorch dataset -----------------------------------------
# Define the dataset class
class MyDataset(Dataset):  # the input is a pytorch built-in Dataset class
        def __init__(self, X, y): # input features and label
            self.X = torch.tensor(X, dtype=torch.float32)  # Convert to tensor, so can be used for NN training
            self.y = torch.tensor(y, dtype=torch.float32)  #
            
        # with following function the data can be used in pytorch dataloader (necessary function)
        def __len__(self):
            return len(self.X)  # it Returns first dimension shape[0]
    
        def __getitem__(self, idx):   # returns X[i], y[i] for dataLoader, one sample X[1,5,88,58], Y[1,1,88,58]
            return self.X[idx], self.y[idx]
            
# split the test and training, create dataloader
def prepare_data(features, labels, device, batch_size=32, num_workers=0, seed = 42,
                 frac_train=0.6, frac_skip=0.2, frac_val=0.1,  frac_test=0.1,
                shuffle=False, normalize_x=True, normalize_y=True):
    """
    Prepares the data by splitting into training , validation and testing sets
    Splits are by proportion; if shuffle=False it uses the original order.
    Returns (train_loader, val_loader, test_loader).
    """
    n = len(features)  # total sample numbers 
    # choose whether need to shuffle the data set
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        X, y = features[idx], labels[idx]
    else:
        X, y = features, labels
    # split the train and test data 
    # total data number is 10169
    n_train = int(frac_train * n)
    # n_train = 500
    n_val = int(frac_val * n)
    n_test  = int(frac_test * n)
    n_skip  = n - n_test - n_val - n_train
    
    print("training samples:", n_train)
    print("skipped samples:", n_skip)
    print("validation samples:", n_val)
    print("test samples:", n_test)
    # Train
    X_train, y_train = X[:n_train], y[:n_train]
    # skip
    X_skip, y_skip = X[n_train:n_train+n_skip], y[n_train:n_train+n_skip]
    # val
    X_val, y_val = X[n_train+n_skip:n_train+n_skip+n_val], y[n_train+n_skip:n_train+n_skip+n_val]
    # test
    X_test, y_test = X[n_train+n_skip+n_val:], y[n_train+n_skip+n_val:]
    # fit norm on train data
    norm = {"X_mu": None, "X_sd": None, "y_mu": None, "y_sd": None}
    if normalize_x:
        X_mu, X_sd = _fit_z_norm_X(X_train)
        norm["X_mu"], norm["X_sd"] = X_mu.squeeze().astype(np.float32), X_sd.squeeze().astype(np.float32) # squeeze removes the dimension of 1, so X_mu -> 5
        # apply to all splits (broadcasting)
        X_train = _apply_z_norm_X(X_train, X_mu, X_sd)
        X_val   = _apply_z_norm_X(X_val,   X_mu, X_sd)
        X_test  = _apply_z_norm_X(X_test,  X_mu, X_sd)

    if normalize_y:
        y_mu, y_sd = _fit_z_norm_y(y_train)
        norm["y_mu"], norm["y_sd"] = y_mu, y_sd
        y_train = _apply_z_norm_y(y_train, y_mu, y_sd)
        y_val   = _apply_z_norm_y(y_val,   y_mu, y_sd)
        y_test  = _apply_z_norm_y(y_test,  y_mu, y_sd)
    
    # create tensor dataset
    train_dataset = MyDataset(X_train, y_train)
    val_dataset = MyDataset(X_val, y_val)
    test_dataset  = MyDataset(X_test,  y_test)

    # DataLoaders, use number of workers to be more efficient
    # pin-memory - GPU
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # how many subprocesses PyTorch uses to load batches in parallel - num_workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory = (device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory = (device.type == "cuda"))
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory = (device.type == "cuda"))

    return train_loader, val_loader, test_loader, norm

