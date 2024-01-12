import os 
import json

import pandas as pd 
if __name__=="__main__":
    path = 'v3_linear_svc.json'
    f = open(path,)
    data = json.load(f)
    
    csv_path = '.'.join([path.split(".")[0], 'csv'])
    df = pd.DataFrame(data)
    df.to_csv(csv_path)

