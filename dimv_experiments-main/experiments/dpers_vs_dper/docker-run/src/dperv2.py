import numpy as np
import time
from tqdm import tqdm 

def diag_term(i, X, y, G):
    arr0 = X[:, i]
    arr = arr0[~np.isnan(arr0)]
    y_arr = y[~np.isnan(arr0)]

    _, counts = np.unique(y_arr, return_counts = True)
    ind = np.insert(np.cumsum(counts), 0, 0)

    return sum([(ind[g] - ind[g - 1]) * np.var(arr[y_arr == g - 1]) for
                g in range(1, G + 1)]) / len(y_arr)


def dper(X, y, G):
    '''
    X: input, should be a numpy array
    y: label
    G: number of classes
    output:
    - mus: each row is a class mean
    - S: common covariance matrix of class 1,2,..., G
    '''
    epsilon = 1e-5  # define epsilon to put r down to 0 if r < epsilon
    n, p = X.shape[0], X.shape[1]

    mus = np.array(
        [np.nanmean(X[y == g, :], axis = 0) for g in range(G)]).T  
    # so that each column is the mean of a class

    S = np.diag([diag_term(i, X, y, G) for i in range(p)])
   
    upper_idx = list(zip(*np.triu_indices(p, 1)))

    for (i, j) in tqdm(upper_idx): 
    #for i in tqdm(range(p)):
    #    for j in range(i):
        if (S[i, i] != 0 and S[j, j] != 0): 
            X_ij = X[:, [i, j]]

            # drop rows with NA
            idx = ~np.isnan(X_ij).any(axis = 1)
            X_ij, y_arr = X_ij[idx], y[idx]

            _, counts = np.unique(y_arr, return_counts = True)
            ind = np.insert(np.cumsum(counts), 0, 0)

            m_g = counts

            A = len(y_arr)

            scaled_X_ij = [X_ij[y_arr == g, :] - mus[[i, j], g] for g in range(G)]

            q = lambda g: np.dot(scaled_X_ij[g][:, 0], scaled_X_ij[g][:, 0])
            s11 = sum(map(q, range(G)))
            q = lambda g: np.dot(scaled_X_ij[g][:, 1], scaled_X_ij[g][:, 1])
            s22 = sum(map(q, range(G)))
            d = lambda g: np.dot(scaled_X_ij[g][:, 0], scaled_X_ij[g][:, 1])
            s12 = sum(map(d, range(G)))

            start_solve = time.time()
            B = S[i, i] * S[j, j] * A - s22 * S[i, i] - s11 * S[j, j]
            coefficient = [-A, s12, B, s12 * S[i, i] * S[j, j]]
            r = np.roots(coefficient)

            r = r[abs(np.imag(r)) < epsilon]
            r = np.real(r)
            r[abs(r) < epsilon] = 0

            if len(r) > 1:
                condi_var = S[j, j] - r ** 2 / S[i, i]
                eta = -A * np.log(condi_var) - (S[j, j] - 2 * r / S[i, i] * s12 +
                                                r ** 2 / S[i, i] ** 2 * s11) / condi_var
                # if condi_var <0 then eta = NA. in practice, it's impossible for cov to be negative
                #  therefore, we drop NA elements of eta
                r = r[eta == max(eta[~np.isnan(eta)])]

            if len(r) > 1:
                w = [m_g[g - 1] * np.cov(X_ij[ind[g - 1]:ind[g], ], rowvar = False) for
                     g in range(1, G + 1)]
                w = np.sum(w, axis = 0)
                r = r[np.abs(r - w[0, 1]).argmin()]  # choose r that is w[0,1]

            S[i, j] = S[j, i] = r
        else:
            S[i, j] = S[j, i] = np.nan
    return [mus, S]
