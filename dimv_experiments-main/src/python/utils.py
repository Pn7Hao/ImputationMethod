import numpy as np

def normalization_scaling(X_missing, mus=None, std=None):
    if not isinstance(mus, np.ndarray):
        mus = np.nanmean(X_missing, axis = 0)
    if not isinstance(std, np.ndarray):
        std = np.nanstd(X_missing, axis = 0) 
    std_not_0 = std != 0  
    
    Xscaled = np.zeros_like(X_missing, dtype=np.float64) 
    Xscaled[:, std_not_0] =\
        (X_missing[:, std_not_0] - mus[std_not_0])/ std[std_not_0] 
    return Xscaled, mus, std

def normalization_rescaling(X_scaled, mus, std):
    return X_scaled * std - mus
