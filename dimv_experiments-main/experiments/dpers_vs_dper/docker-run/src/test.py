
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
import time
from datetime import datetime

module_path = os.path.abspath(os.path.join(''))
if module_path not in sys.path:
    sys.path.append(module_path)  
import json 

#from utils import timeit, rmse_loss 
from dpers import dpers 
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
 
def dpers_run(Xtrain, ytrain, Xtest, ytest):
    print("run dpers")
    duration_path = '../data/dpers_vs_dper/'
    if (os.path.isdir(duration_path)==0):
        os.mkdir(duration_path)

    print(Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape)
    
    results = []
    #create missing data 
    for missing_rate in [.1]:
        print("------")
        print("missing rate", missing_rate)

        X_missing = randomly_missing(Xtrain, missing_rate)
        print(X_missing.shape)
            

        #scaling data for dpers
        X_missing_scaled, mus, std = scaling(X_missing)
        print(X_missing_scaled.shape)
        print(mus.shape)
        print(std.shape)
        #saving Xcaled, mus, std 
        start = time.time() 
        #calc sigma 
        sigma = dpers(X_missing_scaled[:,:])
        duration = time.time() - start 
        print("duration", duration)
        result = {"missing_rate": missing_rate, "duration": duration}
        results.append(result)


    now = datetime.now()
    now_string = now.strftime("%Y%m%d-%H:%M:%S")

     
    
    with open(duration_path+"dpers_ablation_{}.json".format(now_string), "w") as f:
        json.dump(results, f)
        print("complete saving data")

    print("complete Covariance Matrix with DPERS after with result {}".format(results))


def dper_run(Xtrain, ytrain, Xtest, ytest):
    print("run dper")
    duration_path = '../data/dpers_vs_dper/'
    if (os.path.isdir(duration_path)==0):
        os.mkdir(duration_path)

    print(Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape)
    #create missing data 
    results = []
    for missing_rate in [.1]:
        print("-------")
        print("missing_rate", missing_rate)

        X_missing = randomly_missing(Xtrain, missing_rate)
        y = LabelEncoder().fit_transform(ytrain)
        num_classes = np.unique(y).shape[0]
    
        print(X_missing.shape)
        print(num_classes)
        #pd.DataFrame(X_missing).to_csv(missing_data_path, header=False, index=False)
            
        
        #saving Xcaled, mus, std 
    #    pd.DataFrame(X_missing_scaled).to_csv(scaled_data_path, header=False, index=False) 
    #    pd.DataFrame(mus).to_csv(mus_path, header=False, index=False)
    #    pd.DataFrame(std).to_csv(std_path, header=False, index=False)

    
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
        print("complete saving data")
    print("complete Covariance Matrix with DPER after hours with results {} ".format(results))


if __name__=='__main__':

    Xtrain, ytrain, Xtest, ytest = load_data(dataset_name="fashion_mnist")
    Xtrain, ytrain, Xtest, ytest  =  Xtrain[1:1000, :], ytrain[1:1000], Xtest[1:100,], ytest[1:100]  
    dper_run(Xtrain, ytrain, Xtest, ytest ) 
    dpers_run(Xtrain, ytrain, Xtest, ytest )
    
            
