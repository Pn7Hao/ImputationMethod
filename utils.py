import sys
import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import numpy.ma as ma
import pandas as pd
import sklearn.neighbors._base
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
import warnings

from missingpy import MissForest

warnings.simplefilter(action='ignore', category=FutureWarning)
import time

from keras.losses import binary_crossentropy as bce_loss
from keras.losses import mse as mse_loss
from gain import *
from ginn import *
from em import *
from mice import *
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from fancyimpute import SoftImpute,BiScaler  
from tqdm import tqdm


def monotone_missing_img(data_input, perc_del, perc, im_width, im_height):
    # MNAR
    perc_width = perc
    perc_height = perc
    # data : pd.DataFrame
    data = data_input.copy().values
    n = data.shape[0]; 
    m = im_width; 
    p = im_width * im_height

    # position (from width and height) of pixel would be deleted 
    from_width  = math.ceil((1-perc_width)*im_width)-1
    from_height = math.ceil((1-perc_height)*im_width)-1

    nan_rows = np.unique(np.sort(np.random.randint(0, n, int(n*perc_del))))
    nan_rows = nan_rows[:, np.newaxis]

    col_idxs = np.arange(p).reshape(-1, m)

    filter_height = np.arange(from_height, im_height) 
    filter_width  = np.arange(from_width, im_width)

    col_idxs = col_idxs[:, filter_width][filter_height,:].reshape(-1)
    #flatten row removed 
    missing_data = data.copy().astype('float64')
    missing_data[nan_rows, col_idxs] = np.nan

    return (missing_data.reshape(n, p), nan_rows.ravel()) 

def randomly_missing_tabular(data_input, perc):
    # MAR
    np.random.seed(123)
    data = data_input.copy()
    h, w = data.shape[0], data.shape[1] 
    n = data.shape[0]*data.shape[1] 
    # print(h,w,n)
    # perc = n_sample / n 
    flattenX = data.to_numpy().reshape(1, n)
    mask = np.random.uniform(0,1, size = n) < perc
    flattenX[:, mask] = np.nan 
    return pd.DataFrame(data=flattenX.reshape(h, w),columns=data.columns) 

def randomly_missing_img(data,perc):
    '''
    perc : percent of each img -> missing pixel / features
    '''
    np.random.seed(123)
    result = data.copy()
    def add_noise_to_image(image, p=0.1):
        """
        Adds random noise to an image by randomly replacing some pixels with NaN.

        Args:
            image (ndarray): The input image to add noise to.
            p (float): The probability of replacing each pixel with NaN.

        Returns:
            ndarray: The image with added noise.
        """
        # Copy the input image to avoid modifying the original array
        noisy_image = np.copy(image)

        # Create a mask of the same shape as the input image
        mask = np.random.uniform(size=image.shape)
        mask_ones = np.ones_like(mask)

        # Replace some pixels in the mask with NaN based on the probability p
        mask_ones[mask < p] = np.nan

        # Apply the mask to the input image to create the noisy image
        noisy_image = noisy_image * mask_ones

        return noisy_image

    for i in range(len(data)):
        result.iloc[i] = add_noise_to_image(data.iloc[i],p=perc)
    
    return result

def drop_random_data(df, column, n, distribution='uniform', **kwargs):
    np.random.seed(123)
    #'uniform', 'standard', 'normal', or 'poisson'. ? 
    if distribution == 'uniform':
        low = 0
        high = df.shape[0] - 1
        indices = np.random.randint(low, high, size=n)

    elif distribution == 'standard':
        # Use a standard normal distribution to generate random indices
        mean = df[column].mean()
        std = df[column].std()

        indices = np.random.normal(loc=mean, scale=std, size=n)
        # Round the indices to the nearest integer

        indices = np.rint(indices).astype(int)

        # Clip the indices to be within the range of valid indices
        indices = np.clip(indices, 0, df.shape[0] - 1)

    else:
        raise ValueError("Invalid distribution: must be 'uniform' or 'standard'.")

    # Set the specified column values to NaN at the selected indices
    for each in indices:
        df.at[each, column] = np.nan

    return df

def apply_random_missing_for_all(df,n):
  for each in df.columns:
    df = drop_random_data(df, each, n, distribution='uniform')
  return df

def missing(data,Type=None,**k):
    # data : pd.DataFrame
    if Type == None :
        print("Please choice missing data type")
    if Type == 'monotone_img':
        result , nan_rows = monotone_missing_img(data_input=data,**k)
        return pd.DataFrame(result, columns=data.columns)
    if Type == 'randomly_tabular':
        return randomly_missing_tabular(data_input=data,**k)
    if Type == 'randomly_img':
        return randomly_missing_img(data=data,**k)

def apply_normalize(df):
    output = df.copy()
    # scaler = MinMaxScaler() #StandardNormalize -> 
    scaler = StandardScaler()
    for each in output.columns:
      output[each] = scaler.fit_transform(output[[each]])
    return output

def covariance_rmse(cov_matrix1, cov_matrix2):
    std1 = np.sqrt(np.diag(cov_matrix1))
    corr_matrix1 = cov_matrix1 / np.outer(std1, std1)

    std2 = np.sqrt(np.diag(cov_matrix2))
    corr_matrix2 = cov_matrix2 / np.outer(std2, std2)
    diff = corr_matrix1 - corr_matrix2
    squared_diff = np.square(diff)

    mean_squared_diff = np.mean(squared_diff)
    rmse = np.sqrt(mean_squared_diff)

    nan_mask = np.isnan(rmse)
    if np.any(nan_mask):
        # Remove NaN elements from rmse and print a message
        nan_indices = np.where(nan_mask)[0]
        print(f"Elements {nan_indices} are NaN and will be removed.")
        rmse = rmse[~nan_mask]

    total_rmse = (1/len(rmse)) * sum(rmse) 
    print("Total : ",total_rmse)
    return total_rmse


def rmse_ignore_nan(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE) between two matrices, ignoring NaN values.

    Parameters:
    y_true (numpy.ndarray): The true values.
    y_pred (numpy.ndarray): The predicted values.

    Returns:
    float: The RMSE between the true and predicted values, ignoring NaN values.
    """
    # Create a mask of valid elements
    y_true_clean = np.nan_to_num(y_true, nan=-1)
    y_pred_clean = np.nan_to_num(y_pred, nan=-1)
    mask = ~(np.isnan(y_true_clean) | np.isnan(y_pred_clean))

    # Extract only valid elements
    y_true_valid = y_true_clean[mask]
    y_pred_valid = y_pred_clean[mask]
    
    # Calculate RMSE on valid elements
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    return rmse

def impute_with_mice(df, n_iterations=1):
    # imputer = IterativeImputer(max_iter=n_iterations, random_state=123)
    # imputed_array = imputer.fit_transform(df)
    # imputed_df = pd.DataFrame(imputed_array, columns=df.columns)

    imputer = MiceImputer()
    imputed_df = pd.DataFrame(imputer.transform(df.to_numpy(),BayesianRidge , n_iterations),columns=df.columns)

    return imputed_df

def impute_with_missforest(df, n_estimators=1):
    df = df.copy()
    imputer = MissForest(n_estimators=n_estimators, max_iter=1,criterion='mse')
    imputed_array = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_array, columns=df.columns)

    return imputed_df

# def soft_impute_als_df(df, rank=None, max_iter=100, lmbda=0.1, tol=1e-4):
#     """
#     Impute missing values in a pandas DataFrame using Soft-Impute algorithm with Alternating Least Squares (ALS).
    
#     Parameters:
#     -----------
#     df : pandas DataFrame
#         DataFrame to be imputed
#     rank : int or None, optional
#         Rank of the low-rank approximation (default is None, which sets it to the minimum of the number of rows and columns in the input data)
#     max_iter : int, optional
#         Maximum number of iterations (default is 100)
#     lmbda : float, optional
#         Regularization parameter (default is 0.1)
#     tol : float, optional
#         Tolerance for convergence (default is 1e-4)
    
#     Returns:
#     --------
#     pandas DataFrame
#         Imputed DataFrame with missing values filled in.
#     """
    
#     A = df.values

#     mask = np.isnan(A)

#     if rank is None:
#         rank = min(A.shape)-1
#     else:
#         rank = min(rank, min(A.shape))-1

#     A_imputed = np.where(mask, 0, A)
    
#     # Perform Soft-Impute with ALS
#     for i in range(max_iter):
#         U, S, Vt = svds(A_imputed, k=rank)
#         S = np.diag(S)
#         A_imputed_new = U @ S @ Vt
#         A_imputed_new[mask] = 0
        
#         if np.linalg.norm(A_imputed_new - A_imputed) / np.linalg.norm(A_imputed) < tol:
#             break
        
#         A_imputed = A_imputed_new
    
#     df_imputed = pd.DataFrame(A_imputed, index=df.index, columns=df.columns)
    
#     return df_imputed

def soft_impute_als_df(df,n_iter=10):

    imputed_df = SoftImpute(max_iters=n_iter).fit_transform(df)
    imputed_df = pd.DataFrame(imputed_df, columns=df.columns)

    return imputed_df

# def impute_with_pca(data):
#     means = data.mean()
#     data = data.fillna(means)

#     pca = PCA()
#     pca.fit(data)

#     transformed_data = pca.transform(data)

#     imputed_data = pca.inverse_transform(transformed_data)

#     imputed_df = pd.DataFrame(imputed_data, columns=data.columns)
#     return imputed_df

def impute_with_pca(df, n_iter=10):
    # https://github.com/terencechow/PCA
    def svdtriplet(X,roww=[],colw=[],ncp=np.inf):
        if hasattr(X,'pop'):
            X=np.array(X,float)
        elif hasattr(X,'shape'):
            X=X.astype(float)
        else:
            print("Please use a numpy array or a list!")
            return
        if roww==[]:
            roww = np.array(np.ones(X.shape[0])/X.shape[0],float)
        else:
            roww = np.array(roww,dtype=float)
        if colw==[]:
            colw = np.array(np.ones(X.shape[1]),float)
        else:
            colw = np.array(colw,dtype=float)

        ncp = min(ncp,X.shape[0]-1,X.shape[1])
        roww /= roww.sum()
        X *= np.sqrt(colw)
        X *= np.sqrt(roww[:,None])
        U = np.array([])
        s = np.array([])
        V = np.array([])

        if X.shape[1]<X.shape[0]:
            U, s, V = np.linalg.svd(X,full_matrices=False)
            V = V.T
            U = U[:,:ncp]
            V = V[:,:ncp]
            if ncp>1:
                mult = np.sign(V.sum(axis=0))
                mult[mult==0]=1
                U *= mult
                V *= mult
            U /= np.sqrt(roww[:,None])
            V /= np.sqrt(colw[:,None])
        else:
            V, s, U = np.linalg.svd(X.T,full_matrices=False)
            U = U.T
            U = U[:,:ncp]
            V = V[:,:ncp]
            mult = np.sign(V.sum(axis=0))
            mult[mult==0]=1	
            V *= mult
            U *= mult
            U /= np.sqrt(roww[:,None])
            V /= np.sqrt(colw[:,None])

        s = s[:min(X.shape[1],X.shape[0]-1)]
        num = s[:ncp] <1e-15
        if num.sum() >=1:
            U[:,num] *= s[num]
            V[:,num] *= s[num]

        return s,U,V

    def imputePCA(X,ncp=2,scale=True,method=['Regularized','EM'],roww=[],coeffridge=1,threshold=1e-6,seed=123,nbinit=1,maxiter=n_iter):

        """
        This will return a list of 2 numpy arrays. 
        The first array is what the data is with imputed values replacing the missing values 
        and the second numpy array is what the data would look like if all values were imputed.
        """
        def impute(X_input,mx,ncp=4,scale=True,method=None,threshold=1e-6,seed=123,init=1,maxiter=n_iter,roww=None,coeffridge=1):
            X = X_input.copy()
            nbiter = 1
            old = np.inf
            objective = 0
            if seed is not None:
                np.random.seed(seed)
            ncp = min(ncp,X.shape[1],X.shape[0]-1)

            means = ma.average(mx,axis=0,weights=roww).data
            Xhat = X - means
            rows = ma.masked_array(np.vstack([roww]*mx.shape[1]).T,mask=mx.mask)
            standardize = np.sqrt(np.nansum(Xhat**2*roww[:,None],axis=0)/rows.sum(axis=0))
            if scale:
                Xhat/=standardize.data
            Xhat[mx.mask]=0
            if init >1:
                Xhat[mx.mask]=np.random.randn(mx.mask.sum())
            recon = Xhat.copy()
            
            if ncp==0: 
                nbiter=0
            while nbiter>0:
                Xhat[mx.mask] = recon[mx.mask]
                if scale:
                    Xhat*=standardize
                Xhat+=means
                means = np.average(Xhat,axis=0,weights=roww)
                Xhat-=means
                standardize = np.sqrt(np.nansum(Xhat**2*roww[:,None],axis=0)/roww.sum())
                if scale:
                    Xhat/=standardize
                s,U,V = svdtriplet(Xhat,roww=roww)
                sigma2 = np.mean(s[ncp:]**2)
                sigma2 = min(sigma2*coeffridge,s[ncp]**2)
                if 'em' in method:
                    sigma2 = 0
                lambdashrinked = (s[:ncp]**2-sigma2)/s[:ncp]
                recon = np.dot(U[:,:ncp]*roww[:,None]*lambdashrinked,V[:,:ncp].T)
                recon /= roww[:,None]
                diff = Xhat-recon
                diff[mx.mask] = 0
                objective = np.sum(diff**2*roww[:,None])
                criterion = abs(1-objective/old)
                old = objective
                nbiter +=1
                if criterion is not None:
                    if criterion < threshold and nbiter > 5:
                        nbiter = 0
                        print ("Stopped after criterion < threshold")
                    if objective < threshold and nbiter > 5:
                        nbiter = 0
                        print ("Stopped after objective < threshold")
                if nbiter > maxiter:
                    nbiter = 0
                    print ("Stopped after " + str(maxiter) + " iterations")
            if scale:
                Xhat*=standardize
            Xhat+=means
            completeObs = X.copy()
            completeObs[mx.mask] = Xhat[mx.mask]
            if scale:
                recon*=standardize
            recon+=means
            result = [completeObs,recon]
            return result

        ### Impute function done now for rest of impute PCA function
        if hasattr(X,'values'):
            X = X.values.astype(float)
        elif hasattr(X,'shape'):
            X = X.astype(float)
        elif hasattr(X,'pop'):
            X = np.array(X,float)
        else:
            print ("X must be a list, pandas or numpy array")
            return
        method = method[0]
        obj = np.inf
        method = method.lower()
        imputed = np.array([])
        if ncp>min(X.shape[0]-2,X.shape[1]-1):
            print ("Stopping, ncp too large")
            return
        if roww == []:
            roww = np.ones(X.shape[0])/X.shape[0]
        else:
            if hasattr(roww,'pop'):
                roww = np.array(roww)
            elif hasattr(roww,'shape'):
                pass
            else:
                "roww is not a list or numpy array!"
                return
        if np.isnan(np.sum(X)):
            mx = ma.masked_array(X,mask=np.isnan(X))

        for i in range(0,nbinit):
            if ~np.isnan(np.sum(X)):
                return X
            if seed != None:
                seed=seed*i
            else:
                seed = None
            imputeit = impute(X,mx=mx,ncp=ncp,scale=scale,method=method,threshold=threshold,seed=123,init=i+1,maxiter=maxiter,roww=roww,coeffridge=coeffridge)
            if np.mean((imputeit[1][~mx.mask]-X[~mx.mask])**2) <obj:
                imputed = imputeit
                obj = np.mean((imputeit[1][~mx.mask]-X[~mx.mask])**2)
        return imputed

    def estim_ncpPCA(X,ncpmin=0,ncpmax=5,method='regularized',scale=True,cv='gcv',nbsim=100,pNA=0.05,threshold=1e-4):

        if hasattr(X,'values'):
            X = X.values.astype(float)
        elif hasattr(X,'shape'):
            X = X.astype(float)
        elif hasattr(X,'pop'):
            X = np.array(X,float)
        else:
            print ("X must be a list, pandas or numpy array")
            return
        
        method = method.lower()
        cv = cv.lower()
        Xhat = np.array([],float)
        ncpmax = min(ncpmax,X.shape[1]-1,X.shape[0]-2)
        result = []
        
        if cv=='gcv':
            p = X.shape[1]
            n = X.shape[0]
            if ncpmax is None:
                ncpmax = X.shape[1]-1
            ncpmax = min(X.shape[0]-2,X.shape[1]-1,ncpmax)
            crit = []
            if ncpmin == 0:
                mx = ma.masked_array(X,mask=np.isnan(X))
                crit.append(np.mean((mx - np.hstack([[np.mean(mx,axis=0).data]*X.shape[0]]))**2))


            for q in range(max(ncpmin,1),ncpmax+1):
                rec = imputePCA(X,scale=scale,ncp=q,method=method,maxiter=n_iter)[1]

                crit.append(np.mean(((n*p - mx.mask.sum())*(mx-rec)/((n-1)*p - mx.mask.sum() - q*(n+p-q-1)))**2))
            
            ncp = None
            if np.any(np.ediff1d(crit)>0):
                ncp = np.argmax(np.ediff1d(crit)>0)
            else:
                ncp = np.argmin(crit)
            return [ncp,crit]
            
        if cv =='loo':
            res = []
            for nbaxes in range(ncpmin,ncpmax+1):
                Xhat = ma.masked_array(X,copy=True,mask=np.isnan(X))

                it = np.nditer(X,flags=['multi_index'])
                while not it.finished:
                    if ~np.isnan(X[it.multi_index[0],it.multi_index[1]]):
                        mXNA = ma.masked_array(X,copy=True,mask=np.isnan(X))
                        mXNA.mask[it.multi_index[0],it.multi_index[1]]=True
                        mXNA.data[it.multi_index[0],it.multi_index[1]]=None
                        if nbaxes==0:
                            Xhat[it.multi_index[0],it.multi_index[1]] = ma.mean(mXNA[:,it.multi_index[1]])
                        else:
                            Xhat[it.multi_index[0],it.multi_index[1]] = imputePCA(mXNA.data,ncp=nbaxes,threshold=threshold,method=method,scale=scale)[0][it.multi_index[0],it.multi_index[1]]
                    it.iternext()
                res.append(((Xhat-X)**2).mean())
            result = [np.argmin(res)+ncpmin,res]


        if cv == 'kfold':
            res = np.empty((ncpmax-ncpmin+1,nbsim))
            for sim in range(1,nbsim):
                mXNA = ma.masked_array(X,copy=True,mask=np.isnan(X))
                rowsRandom = np.random.random_integers(0,mXNA.shape[0]-1,mXNA.shape[0])
                colsRandom = np.random.random_integers(0,mXNA.shape[1]-1,mXNA.shape[0])
                mXNA.mask[[rowsRandom,colsRandom]]=True
                mXNA.data[[rowsRandom,colsRandom]]=None
                for nbaxes in range(ncpmin,ncpmax+1):
                    if nbaxes==0:
                        Xhat=mXNA.filled(mXNA.mean(axis=0))
                    else:
                        Xhat = imputePCA(mXNA.data,ncp=nbaxes,threshold=threshold,method=method,scale=scale)[0]
                    res[nbaxes-ncpmin,sim] = np.nansum((Xhat-X)**2)
            resmeans = res.mean(axis=1)
            result = [np.argmin(resmeans)+ncpmin,resmeans]
        return result

    q = estim_ncpPCA(df,scale=False)[0]
    pca = imputePCA(df,ncp=q, scale=False,method=['Regularized'],seed=123)
    imputed_df = pd.DataFrame(pca[0], columns=df.columns) 

    return imputed_df

def impute_with_knn(df,n_neighbors=5):
    imp = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    imputed_data = imp.fit_transform(df.to_numpy())
    return pd.DataFrame(imputed_data, columns=df.columns)

def diag_term(i, X, y, G):
    arr0 = X[:, i]
    arr = arr0[~np.isnan(arr0)]
    y_arr = y[~np.isnan(arr0)]

    _, counts = np.unique(y_arr, return_counts = True)
    ind = np.insert(np.cumsum(counts), 0, 0)

    return sum([(ind[g] - ind[g - 1]) * np.var(arr[y_arr == g - 1]) for
                g in range(1, G + 1)]) / len(y_arr)

def to_dper(X, y, G):
    '''
    X: input, should be a numpy array
    y: label
    G: number of classes
    output:
    - mus: each row is a class mean
    - S: common covariance matrix of class 1,2,..., G
    '''
    epsilon = 1e-8  # define epsilon to put r down to 0 if r < epsilon
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

def dper(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    G = len(np.unique(y)) -1 # execpt nan =))

    X = X.values
    y = y.values
    # print(X.shape,y.shape,G)
    mus, S = to_dper(X, y, G)
    return [mus, S]

def dpers(X_input:np.ndarray)->np.ndarray:
    """
    DPER implementations in pythons
    """
    assert isinstance(X_input, np.ndarray) and (np.ndim(X_input) == 2), ValueError("Expected 2D numpy array");
    X = X_input.copy()
    n, p = X.shape;

    # Covariance matrix to be estimated
    S = np.zeros((p, p));
    
    # The diagonal line of S is the sample variance
    for i in range(p):
        x_i = X[:, i];
        x_i = x_i[~np.isnan(x_i)]
        S[i, i] = np.var(x_i) 

    # Upper triangle indices
    upper_idx = list(zip(*np.triu_indices(p, 1)));


    # Calculate the upper triangle matrix of S
    for (i, j) in tqdm(upper_idx):
        X_ij = X[:, [i, j]];
        
        # remove entry with all missing value
        # missing_idx = np.isnan(X_ij).all(1)
        missing_idx = np.isnan(X_ij).any(axis=1)
        X_ij = X_ij[~missing_idx];

        # S_ii, S_jj = S[i, i], S[j, j];
        # if (S_ii != 0) and (S_jj !=0 ):
        #     S[i, j] = find_cov_ij(X_ij, S_ii, S_jj);
        # else:
        #     S[i, j] = np.nan;

        if X_ij.shape[0] > 0:  # check if there are any valid values left
            S_ii, S_jj = S[i, i], S[j, j]
            if (S_ii != 0) and (S_jj != 0):
                S[i, j] = find_cov_ij(X_ij, S_ii, S_jj)
            else:
                S[i, j] = np.nan
        else:
            S[i, j] = np.nan

    S = S + S.T;

    # Halving the diagonal line;
    for i in range(p):
        S[i,i] = S[i,i] * .5;

    return S;
    
def corr_with_dpers(df):
    corr_dpers = dpers(df.to_numpy())
    
    corr_dpers = pd.DataFrame(data=corr_dpers,columns=df.columns)    
    corr_dpers = corr_dpers.set_index(df.columns)
    return corr_dpers


def find_cov_ij(X_ij:np.ndarray, S_ii:float, S_jj:float)->float:
    """
    Given matrix (N x 2), each columns is an observation with some missing value;
    """

    # Column vectors
    # x_i = np.ma.array(X_ij[:, 0], mask = np.isnan(X_ij[:, 0]));
    # x_j = np.ma.array(X_ij[:, 1], mask = np.isnan(X_ij[:, 1]));

    # s11 = np.ma.dot(x_i, x_i) / np.sum(~np.isnan(x_i));
    # s12 = np.ma.dot(x_i, x_j) / np.sum(~np.isnan(x_i * x_j));
    # s22 = np.ma.dot(x_j, x_j) / np.sum(~np.isnan(x_j));
    

    # Number of entries without any missing value
    idx = ~np.isnan(X_ij).any(-1);

    # X without any missing observations
    complt_X = X_ij[idx, :]
    
    # Number of observations without any missing value
    m = np.sum(idx);

    s11 = np.sum(complt_X[:, 0]**2);
    s22 = np.sum(complt_X[:, 1]**2);
    s12 = np.sum(complt_X[:, 0] * complt_X[:, 1]);


    # Coef of polynomial
    coef = np.array([
        s12 * S_ii * S_jj,
        m * S_ii * S_jj - s22 * S_ii - s11 * S_jj,
        s12,
        -m
        ])[::-1]

    roots = np.roots(coef);
    roots = np.real(roots);

    scond = S_jj - roots ** 2/ S_ii;

    # def eta(root):
    #     scond = S_jj - root**2 / S_ii; 
    #     if scond < 0:
    #         return np.NINF;

    #     eta = -m * np.log(scond) - (S_jj - 2 * root / S_ii * s12 + root**2 * s11) / scond;
    #     return eta

    # etas = np.array([eta(root) for root in roots]);
    etas = -m * np.log(scond, out=np.ones_like(scond)*np.NINF, where=(scond>0)) - (S_jj - 2 * roots / S_ii * s12 + roots**2 / S_ii**2 * s11)/scond
    return roots[np.argmax(etas)];

def impute_with_EM(df,n_iter=10):
    # imputer = SimpleImputer(strategy='constant', fill_value=0)
    # imputed_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    imputed_df = EM(max_iter=n_iter,theta=-1e-9).complete(df)
    imputed_df = pd.DataFrame(imputed_df,columns=df.columns)

    return imputed_df

def impute_with_meanImpute(df):
    imputer = SimpleImputer(strategy='mean')
    imputed_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return imputed_df

def gain_impute(df, hint_probability=0.9, batch_size=16, epochs=10):
    data = df.copy()
    # Convert input data to NumPy array
    tmp = data.columns
    data = data.values.astype('float')

    # Create mask of missing values
    mask = np.isnan(data)
    # mask = data.isna().to_numpy()

    # Create generator and discriminator models
    generator = build_generator(data.shape[1])
    discriminator = build_discriminator(data.shape[1])

    # Compile discriminator model
    discriminator.compile(loss='binary_crossentropy', optimizer='adam')

    # Create scaler object for data scaling and unscaling
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data)

    # Scale data to range [-1, 1]
    data_scaled = scaler.transform(data)

    # Train generator and discriminator models
    for epoch in range(epochs):
        # Create batches of observed and missing data
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        for i in range(0, data.shape[0], batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_missing = data_scaled[batch_indices] * mask[batch_indices]
            batch_observed = data_scaled[batch_indices] * (1 - mask[batch_indices])
            batch_hints = None
            batch_hints = (np.random.random(batch_missing.shape) < hint_probability).astype('float')
            batch_hints = tf.cast(batch_hints, tf.float)
            # print(batch_hints.dtype)
            
            # Generate fake data using generator model
            #print('-----------',batch_missing,'========',batch_hints)
            fake_data = generator.predict([batch_missing, batch_hints])
            data_hints = batch_missing * (1 - batch_hints) + fake_data * batch_hints
            
            # Train discriminator model
            discriminator.train_on_batch(batch_observed, np.ones((len(batch_observed), 1)))
            discriminator.train_on_batch(fake_data, np.zeros((len(fake_data), 1)))
            discriminator.train_on_batch(data_hints, batch_hints)

            # Train generator model
            # gen_loss = 0.5 * mse_loss(batch_hints * fake_data,batch_hints * data_scaled[batch_indices]) + \
            #            0.5 * mse_loss((1 - batch_hints) * fake_data, (1 - batch_hints) * batch_observed) + \
            #            0.5 * bce_loss(discriminator(data_hints), tf.cast(batch_hints, dtype='float'))
            generator.train_on_batch([batch_missing, batch_hints], batch_observed)
            
            # Test the generator model by generating some data
            # test_missing = np.zeros((10, data.shape[1]), dtype='float')
            # test_hints = np.ones((10, data.shape[1]), dtype='float')
            # test_data = generator.predict([test_missing, test_hints])
            #print(test_data)
            
    # Generate imputed data using generator model
    imputed_data = generator.predict([data_scaled * mask, np.ones_like(data_scaled)])
    #print([data_scaled * mask])
    #print(imputed_data)
    # Unscale imputed data to original range
    imputed_data = scaler.inverse_transform(imputed_data)

    # Fill in missing values with imputed data
    data_filled = data.copy()
    data_filled[mask] = imputed_data[mask]
    data_filled = np.nan_to_num(data_filled, nan=data_filled.mean())
    data[mask] = imputed_data[mask]
    data = np.nan_to_num(data, nan=data.mean())

    # Convert imputed data to pandas DataFrame
    imputed_df = pd.DataFrame(data_filled, columns=tmp)
    
    # Assuming 'generator' is the generator model object
    # generator_weights = generator.get_weights()
    #print('------------------',generator_weights)

    # return imputed_df,generator
    return imputed_df

def gain_imputed_v2(df, hint_rate=0.9, batch_size=64, alpha=0.5, iterations=10_000):
  '''Impute missing values in data_x
  
  Args:
    - data_x: original data with missing values
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations
      
  Returns:
    - imputed_data: imputed data
  '''
  # Define mask matrix
  data_x = df.to_numpy().copy()
  data_m = 1-np.isnan(data_x)
  
  # System parameters
    #   batch_size = gain_parameters['batch_size']
    #   hint_rate = gain_parameters['hint_rate']
    #   alpha = gain_parameters['alpha']
    #   iterations = gain_parameters['iterations']
  
  # Other parameters
  no, dim = data_x.shape
  
  # Hidden state dimensions
  h_dim = int(dim)
  
  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)
  
  ## GAIN architecture   
  # Input placeholders
  # Data vector
  X = tf.placeholder(tf.float32, shape = [None, dim])
  # Mask vector 
  M = tf.placeholder(tf.float32, shape = [None, dim])
  # Hint vector
  H = tf.placeholder(tf.float32, shape = [None, dim])
  
  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W3 = tf.Variable(xavier_init([h_dim, dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  
  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W3 = tf.Variable(xavier_init([h_dim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
  ## GAIN functions
  # Generator
  def generator(x,m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
    return G_prob
      
  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  
  ## GAIN structure
  # Generator
  G_sample = generator(X, M)
 
  # Combine with observed data
  Hat_X = X * M + G_sample * (1-M)
  
  # Discriminator
  D_prob = discriminator(Hat_X, H)
  
  ## GAIN loss
  D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                + (1-M) * tf.log(1. - D_prob + 1e-8)) 
  
  G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
  
  MSE_loss = \
  tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
  
  D_loss = D_loss_temp
  G_loss = G_loss_temp + alpha * MSE_loss 
  
  ## GAIN solver
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  
  ## Iterations
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
   
  # Start Iterations
  for it in tqdm(range(iterations)):    
      
    # Sample batch
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = norm_data_x[batch_idx, :]  
    M_mb = data_m[batch_idx, :]  
    # Sample random vectors  
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim) 
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp
      
    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
    _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                              feed_dict = {M: M_mb, X: X_mb, H: H_mb})
    _, G_loss_curr, MSE_loss_curr = \
    sess.run([G_solver, G_loss_temp, MSE_loss],
             feed_dict = {X: X_mb, M: M_mb, H: H_mb})
            
  ## Return imputed data      
  Z_mb = uniform_sampler(0, 0.01, no, dim) 
  M_mb = data_m
  X_mb = norm_data_x          
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
  imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
  
  imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
  
  # Renormalization
  imputed_data = renormalization(imputed_data, norm_parameters)  
  
  # Rounding
  imputed_data = rounding(imputed_data, data_x)  
          
  return pd.DataFrame(imputed_data,columns=df.columns)

def get_cat_num_cols(df):
    """
    Returns two lists of column indices for categorical and numerical columns in a pandas DataFrame.
    """
    cat_cols = []
    num_cols = []
    
    for i, col in enumerate(df.columns):
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(i)
        else:
            cat_cols.append(i)
    
    return cat_cols, num_cols

def ginn_imputate(df, perc,Type,epochs=10):
    seed = 123

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    cat_cols,num_cols = get_cat_num_cols(df)
    y = np.reshape(y,-1)
    num_classes = len(np.unique(y))

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y
    )
    # gen randomly missing 
    # cx_train, cx_train_mask = degrade_dataset(x_train, missingness,seed, np.nan)
    # print('>>>>>>>>>')
    # print(cx_train)
    # print("<<<<<<<<< \n",cx_train_mask)
    # cx_test,  cx_test_mask  = degrade_dataset(x_test, missingness,seed, np.nan)
    missing_param = None
    if Type == 'randomly_img' or Type == 'randomly_tabular':
        missing_param = dict(perc=perc)
    elif Type == 'monotone_img':
        missing_param = dict(perc_del=0.5, 
                             perc=perc, 
                             im_width=28, im_height=28) # must custom here
    
    cx_train  = missing(data=x_train,Type=Type,**missing_param).to_numpy()
    cx_train_mask = np.where(np.isnan(cx_train), 0, 1)
    cx_test  = missing(data=x_test,Type=Type,**missing_param).to_numpy()
    cx_test_mask = np.where(np.isnan(cx_test), 0, 1)

    cx_tr = np.c_[cx_train, y_train]
    cx_te = np.c_[cx_test, y_test]

    mask_tr = np.c_[cx_train_mask, np.ones(y_train.shape)]
    mask_te = np.c_[cx_test_mask,  np.ones(y_test.shape)]

    [oh_x, oh_mask, oh_num_mask, oh_cat_mask, oh_cat_cols] = data2onehot(
        np.r_[cx_tr,cx_te], np.r_[mask_tr,mask_te], num_cols, cat_cols)

    oh_x_tr = oh_x[:x_train.shape[0],:]
    oh_x_te = oh_x[x_train.shape[0]:,:]

    oh_mask_tr = oh_mask[:x_train.shape[0],:]
    oh_num_mask_tr = oh_mask[:x_train.shape[0],:]
    oh_cat_mask_tr = oh_mask[:x_train.shape[0],:]

    oh_mask_te = oh_mask[x_train.shape[0]:,:]
    oh_num_mask_te = oh_mask[x_train.shape[0]:,:]
    oh_cat_mask_te = oh_mask[x_train.shape[0]:,:]

    scaler_tr = StandardScaler()
    oh_x_tr = scaler_tr.fit_transform(oh_x_tr)

    scaler_te = StandardScaler()
    oh_x_te = scaler_te.fit_transform(oh_x_te)

    imputer = GINN(oh_x_tr,
               oh_mask_tr,
               oh_num_mask_tr,
               oh_cat_mask_tr,
               oh_cat_cols,
               num_cols,
               cat_cols
              )

    imputer.fit(epochs=epochs)
    imputed_tr = scaler_tr.inverse_transform(imputer.transform())

    imputer.add_data(oh_x_te,oh_mask_te,oh_num_mask_te,oh_cat_mask_te) 
    imputed_te = imputer.transform() 
    imputed_te = scaler_te.inverse_transform(imputed_te[x_train.shape[0]:])

    impute_df = np.concatenate([imputed_tr,imputed_te])
    normal_impute_df = proper_onehot(impute_df,oh_cat_cols)

    impute_ginn_df = pd.DataFrame(data=normal_impute_df,columns=df.columns)
    return impute_ginn_df
