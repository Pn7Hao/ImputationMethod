# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse 
import os
import numpy as np
import pandas as pd 
import json 
from datetime import datetime 
import time 
from xgboost import XGBClassifier 
import json 
from sklearn import metrics

#from load_data import load_data

DATA_FOLDER = "data/mnist/imputed/v13"
ORI_DATA_FOLDER = "data/mnist/original"
SAVED_FOLDER = 'xgboost_gain_20230117' 



get_imputed_data_path = lambda train_or_test, sub_folder, algo: \
        os.path.join(DATA_FOLDER, sub_folder, '{}_{}.csv.gz'.format(train_or_test, algo))
get_y_data_path = lambda train_or_test: \
        os.path.join(ORI_DATA_FOLDER, 'y_{}.csv.gz'.format(train_or_test))



def open_gzip_file(file_path):
    f = open(file_path)
    df = pd.read_table(f, header=0, sep=" ", dtype=np.float64, skiprows = 0)
    return df 

def main(args): 
    '''
    Calculate RMSE of original data set and the imputed one (after rescaled) for all the imputation sample 
    Args:
        - algo_name: algorithm name to calculate the rmse
        - train_or_test: rmse was calcualted on train set or test set 
    Return: 
        - rmse
    '''
    accuracies = {}

    algo_name = args.algo_name

    folders =  [fd for fd in os.listdir(DATA_FOLDER) if fd.split('_')[1] == '10']
    #folders = ['threshold_50_deletedWidthHeightPc_5050_noImagePc_50']
    
    for folder_name in folders:
        print(folder_name)
        data_path = os.path.join(DATA_FOLDER, folder_name)
        if os.path.isdir(data_path):
            ##--------------------------------
            if algo_name =="Gain":
                Xtrain = pd.read_csv(
                        get_imputed_data_path(
                            "train",
                            folder_name, 
                            algo_name
                            ), compression = "gzip"
                        ).to_numpy()
                Xtest = pd.read_csv(
                        get_imputed_data_path(
                            "test",
                            folder_name, 
                            algo_name
                            ), compression = "gzip"
                        ).to_numpy()

            else:
                Xtrain = open_gzip_file(
                    get_imputed_data_path(
                        "train", 
                        folder_name, 
                        algo_name)
                    ).to_numpy()

                Xtest  = open_gzip_file(
                    get_imputed_data_path("test" , folder_name, algo_name)
                    ).to_numpy()

            ytrain = pd.read_csv(
                    get_y_data_path("train"), 
                    compression = "gzip"
                    ).to_numpy().ravel()

            ytest  = pd.read_csv(
                    get_y_data_path("test" ), 
                    compression = "gzip"
                    ).to_numpy().ravel()


            ##--------------------------------
           # Xtrain = pd.read_csv(get_imputed_data_path("train", folder_name, algo_name)).to_numpy()[:5000, ]
           # Xtest  = pd.read_csv(get_imputed_data_path("test" , folder_name, algo_name)).to_numpy()[:1000, ]

           # ytrain = pd.read_csv(get_y_data_path("train", folder_name)).to_numpy().ravel()[:5000, ]
           # ytest  = pd.read_csv(get_y_data_path("test" , folder_name)).to_numpy().ravel()[:1000, ]
            ##--------------------------------

            start = time.time()
            model = XGBClassifier() 
            model.fit(Xtrain, ytrain) 

            ypred = model.predict(Xtest)
            
            acc = metrics.accuracy_score(ytest, ypred)
            print(round(time.time()-start)/60, 3)
            
            accuracies.update({folder_name: acc})

    #---------
    now = datetime.now()
    time_string  = now.strftime("%Y%m%d_%H-%M-%S")
    #---------
 
    print(accuracies)
    saved_path = os.path.join(DATA_FOLDER, "_acc_{}_{}.json".format(
        algo_name, 
        time_string))

    print("complete save result at {}".format(saved_path))
    with open(saved_path, "w") as f:
        json.dump(accuracies, f)
    
    return accuracies

    
if __name__=='__main__':
    #   Inputs for the main function
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--algo_name',
        choices=['impDi', 'softImpute', 'Gain', "knn"],
        default='Gain',
        type=str)
   
    args = parser.parse_args() 
    
    rmses = main(args)
