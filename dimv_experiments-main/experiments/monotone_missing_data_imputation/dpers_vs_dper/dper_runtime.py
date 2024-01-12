import os
import sys 
import pandas as pd
import numpy as np
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

y = pd.read_csv(os.path.join(data_path, "labels.csv"))
y = y[['Class']].to_numpy().ravel()
le2 = LabelEncoder()
y = le2.fit_transform(y)  


def randomly_missing(data, perc):
    h, w = data.shape[0], data.shape[1] 
    n = data.shape[0]*data.shape[1] 
    flattenX = data.reshape(1, n)
    mask = np.random.uniform(0,1, size = n) < perc
    flattenX[:, mask] = float("NaN")    
    return flattenX.reshape(h, w) 
 
if __name__=='__main__':
    X_missing = randomly_missing(X, 0.2)
    start = time.time()
    mus, sigma = dper(X_missing[:, :], y, 5)
     
    running_time = time.time() - start 
    print("dpers running time (in second)", running_time)
    sigma_df = pd.DataFrame(sigma)
    sigma_path="../../data/gen_expression_cancer/sigma/dper.csv"
    sigma_time_path = "../../data/gen_expression_cancer/time/dper.txt"


    sigma_df.to_csv(sigma_path)
    f = open(sigma_time_path, "a")
    f.write(str(running_time))
    f.close()
