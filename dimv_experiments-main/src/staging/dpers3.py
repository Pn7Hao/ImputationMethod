import os
import sys 
import pandas as pd
import numpy as np
from tqdm import tqdm
 
    
import time 

def dpers(X:np.ndarray)->np.ndarray:
    N, P = X.shape;

    # Covariance matrix to be estimated
    S = np.zeros((P, P));

    # The diagonal line of S is the sample variance
    np.fill_diagonal(S, np.nanvar(X, axis=0)) 

    # Upper triangle indices
    upper_idx = list(zip(*np.triu_indices(P, 1)));
    Index = np.array(upper_idx) 
    
    @np.vectorize 
    def find_cov_upper_triag(i, pbar):
        pbar.update(1)
        if (S[Index[i, 0], Index[i, 0]] == 0 \
            or S[Index[i, 1], Index[i, 1]] == 0):
            return np.nan
        
        return find_cov_ij(
                    X[:,[Index[i,0],Index[i,1]]] , 
                    S[Index[i,0], Index[i,0]], 
                    S[Index[i,1], Index[i,1]]
                ) 
    
    with tqdm(total=len(Index)) as pbar:  
        S_upper = find_cov_upper_triag(np.arange(len(Index)), pbar)
    
    assert (len(S_upper) == len(S[np.triu_indices(P,1)])), "Output of S's upper triangle is not in a correct shape" 
    S[np.triu_indices(P, 1)] = S_upper  
    S = S + S.T 
    np.fill_diagonal(S, np.diag(S)*0.5) 
    return S 

def find_cov_ij(Xij, Sii, Sjj): 
    df = pd.DataFrame(Xij)  
    df.dropna(axis = 0, how = 'any', inplace = True)
    comlt_Xij = df.to_numpy()

    s11 = sum(comlt_Xij[:,0]**2)
    s12 = sum(comlt_Xij[:,0]*comlt_Xij[:,1])
    s22 = sum(comlt_Xij[:,1]**2) 

    m = df.shape[0]
    coef = [
        s12*Sii*Sjj, 
        m*Sii*Sjj-s22*Sii-s11*Sjj,
        s12, 
        -m ][::-1] 

    roots = np.roots(coef);
    roots = np.real(roots);

    scond = Sjj - (roots ** 2)/ Sii; 
    etas = -m * np.log(scond, out=np.ones_like(scond)*np.NINF, where=(scond>0)) - (Sjj-2*roots/Sii*s12+roots**2/Sii**2*s11)/scond 
    return roots[np.argmax(etas)]  
