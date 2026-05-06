import numpy as np

if __name__ == "__main__":
    #  load the data
    path_scratch = '/scratch/jczhang/'
    path_ds = 'submeso_data_driven/src/post_analysis/05_Generalization_test/'
    # U
    data = np.load(path_scratch + path_ds + "U.npz")
    U = data["data"]
    print(U.shape)
    # V
    data = np.load(path_scratch + path_ds + "V.npz")
    V = data["data"]
    print(V.shape)
    # W
    data = np.load(path_scratch + path_ds + "W.npz")
    W = data["data"]
    print(W.shape)
    # vorticity
    data = np.load(path_scratch + path_ds + "zeta.npz")
    zeta = data["data"]
    print(zeta.shape)
    #  strain
    data = np.load(path_scratch + path_ds + "strain.npz")
    strain = data["data"]
    print(strain.shape)
    # potential temperature
    data = np.load(path_scratch + path_ds + "T.npz")
    T = data["data"]
    print(T.shape)
    # speed
    data = np.load(path_scratch + path_ds + "speed.npz")
    speed = data["data"]
    print(speed.shape)
    # kpphbl
    data = np.load(path_scratch + path_ds + "kpphbl.npz")
    kpphbl = data["data"]
    print(kpphbl.shape)
    # submesoscale heat flux
    data = np.load(path_scratch + path_ds + 'wT.npz')
    wT_submeso = data["data"]
    print(wT_submeso.shape)
    
    X = np.stack([U, V, W, zeta, strain, T, speed, kpphbl], axis=1)
    y = np.expand_dims(wT_submeso, axis=1)
    
    print('dimension of all features:', X.shape)
    print('dimension of label:', y.shape)

    # check if there is nan value
    nan_count = np.sum(np.isnan(X))
    print(f"Number of NaN values in feature: {nan_count}")
    nan_count = np.sum(np.isnan(y))
    print(f"Number of NaN values in label: {nan_count}")

    np.save("./" + "X.npy", X)
    np.save("./" + "y.npy", y)
    print('dataset is saved')