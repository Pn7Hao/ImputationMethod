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


def randomly_missing(data, perc):
    h, w = data.shape[0], data.shape[1] 
    n = data.shape[0]*data.shape[1] 
    flattenX = data.reshape(1, n)
    mask = np.random.uniform(0,1, size = n) < perc
    flattenX[:, mask] = float("NaN")    
    return flattenX.reshape(h, w) 
 
if __name__=='__main__':
    start = time.time()
    saved_path = "../../data/gen_expression_cancer/missing/data.csv" 
    X_missing = randomly_missing(X, 0.2)
    df_missing = pd.DataFrame(X_missing)
    
    df_missing.to_csv(saved_path)
    print("saving after", time.time()-start)

