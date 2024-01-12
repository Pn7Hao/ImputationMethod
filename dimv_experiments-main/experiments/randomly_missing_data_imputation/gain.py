# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Main function for Mnist dataset '''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os 
import numpy as np
import pandas as pd 

from gain import gain

ROOT = "../data/randomly_missing_dataset/svmRadial_20230214"
DATASETS =  ["iris", "ionosphere", "seeds", "wine", "breast_tissue", "parkinsons", "new_thyroid", "breast_cancer_wisconsin"] 

get_dataset_folders = lambda dataset: [f for f in os.listdir(os.path.join(ROOT, dataset)) if f.split("_")[0]=="missing"]
get_folds = lambda dataset, folder: [f for f in os.path.join(ROOT, dataset, folder) if f in range(10)] 


get_Xmissing_path = lambda dataset_name, folder_name, fold_number, train_or_test: \
       os.path.join(ROOT, "{}/{}/{}/missing_X_{}_normed.csv".format(
           dataset_name, folder_name, fold_number, train_or_test
           ))

get_Xori_path = lambda dataset_name, folder_name, fold_number, train_or_test: \
       os.path.join(ROOT, "{}/{}/{}/ori_X_{}.csv".format(
           dataset_name, folder_name, fold_number, train_or_test
           ))
get_y_path = lambda dataset_name, folder_name, fold_number, train_or_test: \
       os.path.join(ROOT, "{}/{}/{}/y_{}.csv".format(
           dataset_name, folder_name, fold_number, train_or_test
           ))




def main (args):
    print("running")
    '''Main function for UCI letter and spam datasets.
    
    Args:
      - data_name: letter or spam
      - miss_rate: probability of missing components
      - batch:size: batch size
      - hint_rate: hint rate
      - alpha: hyperparameter
      - iterations: iterations
      
    Returns:
      - imputed_data_x: imputed data
      - rmse: Root Mean Squared Error
    '''
    
    
    gain_parameters = {'batch_size': args.batch_size,
                       'hint_rate': args.hint_rate,
                       'alpha': args.alpha, 
                       'iterations': args.iterations}


    for dataset in DATASETS:
        print("dataset")
        for folder_name in get_dataset_folders(dataset):
            print(dataset, folder_name)
            for fold in get_folds(dataset, folder):
                data_path = os.path.join(ROOT, dataset, folder_name, fold)
                missing_X_train  = pd.read_csv(get_Xmissing_path(dataset, folder, fold, "train"))
                missing_X_test   = pd.read_csv(get_Xmissing_path(dataset, folder, fold, "test"))

                ori_X_train  = pd.read_csv(get_Xori_path(dataset, folder, fold, "train"))
                ori_X_test  = pd.read_csv(get_Xori_path(dataset, folder, fold, "test"))

                y_train  = pd.read_csv(get_y_path(dataset, folder, fold, "train"))
                y_test  = pd.read_csv(get_y_path(dataset, folder, fold, "test"))



                # Impute missing data
                X_train_imputed = gain(missing_X_train, gain_parameters)
                X_test_imputed  = gain(missing_X_test, gain_parameters)
                print(X_train_imputed.shape)
                print(X_test_imputed.shape)


                test_Gain = pd.DataFrame(X_train_imputed)
                test_Gain.to_csv(os.path.join(data_path, "train_Gain.csv"), index=False)

                train_Gain = pd.DataFrame(X_test_imputed)
                train_Gain.to_csv(os.path.join(data_path, "test_Gain.csv"),index=False)
                print("Compele save imputation")
        

if   __name__ == '__main__':  
     
#   Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=128,
        type=int)
    parser.add_argument(
        '--hint_rate',
        help='hint probability',
        default=0.9,
        type=float)
    parser.add_argument(
        '--alpha',
        help='hyperparameter',
        default=100,
        type=float)
    parser.add_argument(
        '--iterations',
        help='number of training interations',
        default=10000,
        type=int)
    
    args = parser.parse_args() 
    
    # Calls main function  
    mse = main(args)
