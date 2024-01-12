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
from dper import dper 
from load_data import load_data


 
def main():
    print("run")
    Xtrain, ytrain, Xtest, ytest = load_data(dataset_name="fashion_mnist")
    print(Xtrain.shape)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    main()
            

