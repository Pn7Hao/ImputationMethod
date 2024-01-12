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
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)  
    
from src.dpers import dpers 
from src.dper import dper 

data_path = "../../data/gen_expression_cancer/raw"
os.listdir(data_path) 

df = pd.read_csv(os.path.join(data_path, "data.csv"), )
X = df[df.columns[1:]].to_numpy() 

def scaling(X_missing):
    mus = np.nanmean(X_missing, axis = 0)
    std = np.nanstd(X_missing, axis = 0) 
    std_not_0 = std != 0 
    
    Xscaled = np.zeros_like(X_missing) 
    Xscaled[:, std_not_0] =\
        (X_missing[:, std_not_0] - mus[std_not_0])/ std[std_not_0] 
    return Xscaled, mus, std 
 
def randomly_missing(data, perc):
    h, w = data.shape[0], data.shape[1] 
    n = data.shape[0]*data.shape[1] 
    flattenX = data.reshape(1, n)
    mask = np.random.uniform(0,1, size = n) < perc
    flattenX[:, mask] = float("NaN")    
    return flattenX.reshape(h, w) 
 
def dpers_experiment_run(X):
    dper_root = "../../data/gen_expression_cancer/dpers/v1/"
 
    if not os.path.isdir(dper_root):
        os.mkdir(dper_root)
        
    #path to save data 
    sigma_path = os.path.join(dper_root, 'sigma.csv')
    sigma_time_path =  os.path.join(dper_root, 'time.csv')
    missing_data_path = os.path.join(dper_root, 'missing.csv')
    scaled_data_path = os.path.join(dper_root, 'scaled.csv')
    mus_path = os.path.join(dper_root, 'mus.csv')
    std_path = os.path.join(dper_root, 'std.csv')
    

    #create missing data 
    X_missing = randomly_missing(X, 0.2)
    pd.DataFrame(X_missing).to_csv(missing_data_path, header=False, index=False)
        

    #scaling data for dpers
    X_missing_scaled, mus, std = scaling(X_missing)
    #saving Xcaled, mus, std 
    pd.DataFrame(X_missing_scaled).to_csv(scaled_data_path, header=False, index=False)
    pd.DataFrame(mus).to_csv(mus_path, header=False, index=False)
    pd.DataFrame(std).to_csv(std_path, header=False, index=False)

    start = time.time() 
    #calc sigma 
    sigma = dpers(X_missing_scaled[:,:])
    duration = time.time() - start 
    pd.DataFrame(sigma).to_csv(sigma_path, header=False, index=False)
    
    f = open(sigma_time_path, "a")
    f.write(str(duration))
    f.close()
    print("complete Covariance Matrix with DPERS after: {} hours".format(str(duration/60/60)))


if __name__=='__main__':

     dpers_experiment_run(X) 

