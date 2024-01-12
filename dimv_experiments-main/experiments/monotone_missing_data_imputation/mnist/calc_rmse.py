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

DATA_FOLDER = "data/mnist/imputed/v13"
ORI_DATA_FOLDER = "data/mnist/original"

def get_normalization_parameters(X_train_ori, X_train_normed_missing):
    nan_filter = np.isnan(X_train_normed_missing)
    _X_train_ori = X_train_ori.astype(np.float64)
    _X_train_ori[nan_filter] = float("NaN")

    mus = np.nanmean(_X_train_ori, axis = 0)
    stds = np.nanstd(_X_train_ori, axis = 0) 
    
    return mus, stds


def normalization_rescaling(X_scaled, mus, std):
    return X_scaled * std - mus


def rmse_loss(ori_data, imputed_rescaled_data, missing_pos_filter):
    '''Compute RMSE loss between ori_data and rescaled_data 
    Args:
    Return:
    '''
    nominator = np.sum(((missing_pos_filter * ori_data) - (missing_pos_filter * imputed_rescaled_data))**2)
    denominator = np.sum(missing_pos_filter)
 
    rmse = np.sqrt(nominator/float(denominator))
    return rmse 



get_normed_missing_data_path= lambda train_or_test, sub_folder, algo: \
        os.path.join(DATA_FOLDER, sub_folder, 'X_{}_normed.csv'.format(train_or_test, algo))
get_imputed_data_path = lambda train_or_test, sub_folder, algo: \
        os.path.join(DATA_FOLDER, sub_folder, '{}_{}.csv'.format(train_or_test, algo))
get_ori_data_path = lambda train_or_test: \
        os.path.join(ORI_DATA_FOLDER, "X{}.csv".format(train_or_test))
get_rescaled_data_path= lambda train_or_test, sub_folder, algo: \
        os.path.join(DATA_FOLDER, sub_folder, '{}_Gain_Xrecon.csv'.format(train_or_test, algo))


def calc_rmse(algo_name, train_or_test): 
    '''
    Calculate RMSE of original data set and the imputed one (after rescaled) for all the imputation sample 
    Args:
        - algo_name: algorithm name to calculate the rmse
        - train_or_test: rmse was calcualted on train set or test set 
    Return: 
        - rmse
    '''
    rmses = {}
    
    folders =  os.listdir(DATA_FOLDER) 
    #folders = ['threshold_50_deletedWidthHeightPc_5050_noImagePc_50']

    for folder_name in folders:
        data_path = os.path.join(DATA_FOLDER, folder_name)
        if os.path.isdir(data_path):
            X_train_ori = pd.read_csv(get_ori_data_path("train")).to_numpy()
            X_train_normed_missing = pd.read_csv(get_normed_missing_data_path("train", folder_name, algo_name))
            mus, stds = get_normalization_parameters(X_train_ori, X_train_normed_missing)
            del X_train_ori
            del X_train_normed_missing 

            missing_normed_data = pd.read_csv(
                    get_normed_missing_data_path(train_or_test, folder_name, algo_name)).to_numpy()

            imputed_data = pd.read_csv(
                    get_imputed_data_path(train_or_test, folder_name, algo_name)).to_numpy()

            ori_data = pd.read_csv(
                    get_ori_data_path(train_or_test)).to_numpy()

            rescaled_imputed_data = normalization_rescaling(
                imputed_data, mus, stds)
            
            rescaled_df = pd.DataFrame(rescaled_imputed_data)
            rescaled_df.to_csv(get_rescaled_data_path(train_or_test, folder_name, algo_name))

            missing_mask = np.isnan(missing_normed_data)*1
            # calculate rmse 
            rmse = rmse_loss(ori_data, rescaled_imputed_data, missing_mask)
            print(folder_name, rmse)

            rmses.update({folder_name: rmse})
    #---------
    now = datetime.now()
    time_string  = now.strftime("%Y-%m-%d_%H-%M-%S")
    #---------
 
    print(rmses)
    saved_path = os.path.join(DATA_FOLDER, "_rmse_{}_{}.json".format(
        algo_name, 
        time_string))
    print("complete save result at {}".format(saved_path))
    with open(saved_path, "w") as f:
        json.dump(rmses, f)
    
    return rmses
   

def main(args):
    algo_name = args.algo_name 
    
    calc_rmse(algo_name, "train")
    calc_rmse(algo_name, "test")


if __name__=='__main__':
    #   Inputs for the main function
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--algo_name',
        choices=['impDi', 'softImpute', 'Gain'],
        default='impDi',
        type=str)

    args = parser.parse_args() 
    
    rmses = main(args)
