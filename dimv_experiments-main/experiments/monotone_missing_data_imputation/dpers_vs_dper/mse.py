import os
import numpy as np
import pandas as pd 
import json 

def mse(a, b):
    return (np.square(a-b)).mean(axis=None)

def calc_rmse(sub_folder, train_or_test):
    root = '../../data/mnist/imputed/v11'
    
    get_Xpath = lambda train_test, algo: \
            os.path.join(root, sub_folder, '{}_{}_Xrecon.csv'.format(train_or_test, algo))
    
    softImpute_Xpath = get_Xpath(train_or_test, 'softImpute')
    impDi_Xpath = get_Xpath(train_or_test, 'impDi')
    original_Xpath = os.path.join(
            root, 
            "../../processed",
            ''.join(["X", train_or_test, ".csv"])) 
    
    softImpute_imputed = pd.read_csv(softImpute_Xpath).to_numpy()
    impDi_imputed = pd.read_csv(impDi_Xpath).to_numpy()
    original_data = pd.read_csv(original_Xpath).to_numpy() 


    softImputed_mse = mse(softImpute_imputed, original_data)
    impDi_mse = mse(impDi_imputed, original_data) 

    
    result = {train_or_test: {"softImputed_mse": softImputed_mse, "impDi_mse": impDi_mse}} 
    print(result)
    return result 

def calc_rmse_pipeline(sub_folder):
    result = calc_rmse(sub_folder, "train")
    result.update(calc_rmse(sub_folder, "test"))

    result_saved_path = os.path.join(root, sub_folder, 'mse.json')

    with open(result_saved_path, 'w') as f:
        _result = json.dump(result, f) 
    

if __name__=='__main__':
    root = '../../data/mnist/imputed/v11/'
    #print(os.listdir(root))
    for sub_folder in os.listdir(root):
        if len(sub_folder.split("_")) >3 \
                and sub_folder.split("_")[-1]=='50':
            print(sub_folder)
            calc_rmse_pipeline(sub_folder)
