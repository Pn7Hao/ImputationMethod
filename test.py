from utils import *
import numpy  as np
import pandas as pd
import gc

from tqdm import tqdm # 13p 51s
percs = [0.4,0.5,0.6]
for perc in tqdm(percs):
    corr_gt = np.load(f'DATA/corr_gt_{perc}.npy')
    new_df = pd.read_csv(f'DATA/mnist_ms_nm_{perc}.csv')
    imputed = gain_imputed_v2(new_df.drop('label',axis=1),hint_rate=0.9, batch_size=64, iterations=10_000,alpha=0.5)
    corr_GAIN = np.corrcoef(imputed.T)
    # np.save(f'experiments/Corr/MNIST/GAIN/corr_{perc}_v3.npy', imputed)
    np.save(f'experiments/Corr/MNIST/GAIN/corr_{perc}.npy', corr_GAIN)
    np.save(f'experiments/Rmse/MNIST/GAIN/rmse_{perc}.npy',rmse_ignore_nan(corr_gt,corr_GAIN))
