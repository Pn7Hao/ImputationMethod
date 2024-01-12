import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits,load_iris
from sklearn.datasets import fetch_openml
from utils import *
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor
import functools


def main():
    
    def process_algo(params, names, perc):
        print("--------- ",names)
        corr, rmse = Algo(params[0], **params[1])
        np.savetxt(f'{names}_{perc}_corr.txt', corr, fmt='%d')
        #np.savetxt(f'{names}_{perc}_rmse.txt', rmse, fmt='%d')
        # f = open(f'{names}_{perc}_corr.txt','w')
        # f.write(corr)
        # f.close()
        f = open(f'{names}_{perc}_rmse.txt','w')
        f.write(rmse)
        f.close()
        return [rmse,corr]

    def Algo(func,**kwargs):
        impute = func(**kwargs) 
        if impute.shape == kwargs['df'].shape:
            # corr = impute.corr()
            try:
                corr = np.corrcoef(impute.drop('label',axis=1).T)
            except:
                corr = np.corrcoef(impute.T)
        else:
            corr = impute  # in case dpers
        # rmse_cu = covariance_rmse(corr_gt, corr)
        # rmse_moi = rmse_ignore_nan(corr_gt.to_numpy(), corr.to_numpy())
        rmse_moi = rmse_ignore_nan(corr_gt, corr)
        # return corr,rmse_cu,rmse_moi
        return corr,rmse_moi

    # percs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    percs = [0.4]
    # exps = ['Iris', 'Digits','MNIST']
    # Types = ['randomly_tabular','randomly_img','monotone_img']
    exps = ['MNIST']
    Types = ['monotone_img']
    start = time.time()
    
    for exp,tpe in zip(exps,Types):
        pers_verfied = []
        rmse_vis = defaultdict(list)
        for perc in tqdm(percs): 
            # if perc in [0.1, 0.2, 0.3] and tpe == 'monotone_img':
                # continue
            # if perc == 0.6 and tpe in ['randomly_tabular','randomly_img']:
                # continue
            if exp == 'Digits':
            # ---------------- digits----------------------
                digits = load_digits()
                df = pd.DataFrame(digits.data)
                df['target'] = digits.target
                new_df = missing(df, Type=tpe, perc=perc)
                new_df['target'] = digits.target
                new_df = apply_normalize(new_df)
                corr_gt = df.corr()
            # print(sum(new_df.isna().sum()))
            # ---------------- digits----------------------
            elif exp == 'Iris':
            # ---------------- iris ----------------------
                data = load_iris()
                df = pd.DataFrame(data= np.c_[data['data'], data['target']],
                            columns= data['feature_names'] + ['target'])
                new_df = missing(df,Type=tpe,perc=perc)
                n_sample = df.shape[0] * perc
                new_df = apply_normalize(new_df)
                corr_gt = df.corr()
            
            # ---------------- MNIST ----------------------
            elif exp == 'MNIST':
                mnist = fetch_openml('mnist_784')
                # Split the data into features (X) and labels (y)
                X = mnist.data
                y = mnist.target
                df = pd.DataFrame(X) 
                new_df = missing(df,Type=tpe,perc=perc,perc_del=0.5, im_width=28,im_height=28)
                new_df = apply_normalize(new_df)
                df['label'] = y 
                # new_df['label'] = y
                corr_gt = np.corrcoef(df.drop('label',axis=1).T)
            # ---------------- MNIST ----------------------

            all = defaultdict(list)
            Algo_impute = {
                'MICE':             [impute_with_mice,          dict(df=new_df, n_iterations = 1)],
                'MissForest':       [impute_with_missforest,    dict(df=new_df, n_estimators = 1)],
                'ALS':              [soft_impute_als_df,        dict(df=new_df)],
                'PCA':              [impute_with_pca,           dict(df=new_df)],
                'KNN':              [impute_with_knn,           dict(df=new_df)],
                'EM' :              [impute_with_EM,            dict(df=new_df)],
                'MeanImpute' :      [impute_with_meanImpute,    dict(df=new_df)],
                'Dpers':            [corr_with_dpers,           dict(df=new_df)],
                'GAIN':             [gain_impute,               dict(df=new_df, hint_probability=0.9, batch_size=64, epochs=1)],
                'GINN':             [ginn_imputate,             dict(df=apply_normalize(df), perc=perc,Type=tpe, epochs=1)] # GINN have randomly in his mask
            }
            # for names, params in tqdm(Algo_impute.items()):
                # print("--------- ",names)
                # # corr, rmse1,rmse2 = Algo(params[0], **params[1])
                # # all[names].append([[rmse1,rmse2],corr])
                # corr, rmse = Algo(params[0], **params[1])
                # np.savetxt(f'{names}_{perc}_corr.txt', corr, fmt='%d')
                # np.savetxt(f'{names}_{perc}_rmse.txt', rmse, fmt='%d')
                # all[names].append([rmse,corr])
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for names, params in tqdm(Algo_impute.items()):
                    future = executor.submit(functools.partial(process_algo, params, names, perc))
                    futures.append(future)

                results = []
                for future in tqdm(futures):
                    result = future.result()
                    results.append(result)
                    all[names].append(result)
                
            pers_verfied.append(perc)

            # Create a figure with two subplots
            fig, axs = plt.subplots(ncols=5,nrows=2, figsize=(22, 8))
            # Plot each correlation matrix on the appropriate subplot
            for i, each in enumerate(all):
                matrix = all[each][0][1]
                # rmse_vis1[each].append(all[each][0][0][0])
                # rmse_vis2[each].append(all[each][0][0][1])
                rmse_vis[each].append(all[each][0][0])
                row = i // 5
                col = i % 5
                if col == 4:
                    im = sns.heatmap(matrix, cmap='coolwarm', cbar=False, ax=axs[row, col], vmin=-1, vmax=1)
                else:
                    im = sns.heatmap(matrix, cmap='coolwarm', cbar=False, ax=axs[row, col], vmin=-1, vmax=1)
                axs[row, col].set_title(f'{each}')

            cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
            fig.colorbar(im.get_children()[0], cax=cbar_ax)
            # Set the label for the colorbar
            axs[0, 0].set_ylabel('Correlation', fontsize=14)

            # Add a title for the entire figure
            fig.suptitle('Correlation Matrices', fontsize=16)

            # Adjust the spacing between subplots
            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            # Hide the ticks and labels on all subplots
            for ax in axs.flat:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            # Show the figure
            plt.savefig(f'experiments/{exp}_final_{perc}.png',dpi=600)
            # plt.show()
        print(rmse_vis)
        with open('rmse_vis.txt', 'w') as f:
            for key, value in rmse_vis.items():
                f.write(f"{key}: {value}\n")
        
        # print(rmse_vis2)
        fig, ax = plt.subplots(figsize=(22, 15))

        # Loop through each key and value in the dictionary
        for key, value in rmse_vis.items():
            # Plot the line chart
            ax.plot(pers_verfied, value, label=key)

        # for key, value in rmse_vis2.items():
        #     # Plot the line chart
        #     ax.plot(percs, value, label=key+"Moi")

        # Add x-axis and y-axis labels
        ax.set_xlabel('Perc')
        ax.set_ylabel('Values')

        # Add a title
        ax.set_title(f'{exp} RMSE Line Chart')

        # Add a legend
        ax.legend()

        # Show the plot
        plt.savefig(f'experiments/rmse_{exp}.png')
        # plt.show()
        # plt.plot(rmse_vis)
        print('Finished Time : ',time.time()-start)
if __name__ == "__main__":
    
    main()
