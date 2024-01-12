from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse 

import os
import sys 
import pandas as pd
import numpy as np
import pandas as pd 

from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from multiprocessing import Pool
from  sklearn.preprocessing import LabelEncoder  
import requests 
import time
from datetime import datetime 
import json 
module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)  

#from utils import timeit, rmse_loss 
from dperv2 import dper 
from load_data import load_data


def scaling(X_missing):
    X = X_missing.copy()
    mus = np.nanmean(X, axis = 0)
    std = np.nanstd(X, axis = 0) 
    std_not_0 = std != 0 
    
    Xscaled = np.zeros_like(X) 
    Xscaled[:, std_not_0] =\
        (X[:, std_not_0] - mus[std_not_0])/ std[std_not_0] 
    return Xscaled, mus, std 


def randomly_missing(datasource, perc):
    data = datasource.copy()
    data = data.astype(float)
    h, w = data.shape[0], data.shape[1] 
    n = data.shape[0]*data.shape[1] 
    flattenX = data.reshape(1, n)
    mask = np.random.uniform(0,1, size = n) < perc
    flattenX[:, mask] = float("NaN")    
    return flattenX.reshape(h, w) 


def main():
    Xtrain, ytrain, Xtest, ytest = load_data(dataset_name="fashion_mnist")
    print(Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape)
    #create missing data 
    results = []
    for missing_rate in [.1, .2, .3, .4, .5, .6, .7,.8, .9]:
        print("-------")
        print("missing_rate", missing_rate)

        X_missing = randomly_missing(Xtrain, missing_rate)
        y = LabelEncoder().fit_transform(ytrain)
        num_classes = np.unique(y).shape[0]
    
        print(X_missing.shape)
        print(y)
        print(num_classes)
        #pd.DataFrame(X_missing).to_csv(missing_data_path, header=False, index=False)
            
        
        #saving Xcaled, mus, std 
    #    pd.DataFrame(X_missing_scaled).to_csv(scaled_data_path, header=False, index=False) 
    #    pd.DataFrame(mus).to_csv(mus_path, header=False, index=False)
    #    pd.DataFrame(std).to_csv(std_path, header=False, index=False)
        duration_path = 'data/dpers_vs_dper/'
        if (os.path.isdir(duration_path)==0):
            os.mkdir(duration_path) 
    
        start = time.time() 
        #calc sigma 
        sigma = dper(X_missing[:,:], y, num_classes )
        duration = time.time() - start 
        #pd.DataFrame(sigma).to_csv(sigma_path, header=False, index=False)
        result = {"missing_rate": missing_rate, "duration": duration}
        results.append(result)

    now = datetime.now()
    now_string = now.strftime("%Y%m%d - %H:%M")
    
    file_path = duration_path+"dper_ablation_{}.json".format(now_string) 
    print("file_path: ", file_path)
    with open(file_path, "w") as f:
        json.dump(results, f)
    print("complete Covariance Matrix with DPER after hours with results {} ".format(results))

if __name__=='__main__':
    main()
            

