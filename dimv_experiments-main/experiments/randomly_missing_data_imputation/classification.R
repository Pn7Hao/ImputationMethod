## ----setup, include = FALSE--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(cache = TRUE, echo=TRUE, eval = TRUE)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
require(knitr)
purl("classification.Rmd", output = 'classification.R')


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

DATASETS = c("yeast", "iris", "new_thyroid")#c("wine_quality", "soybean", "ecoli")

FUNC_LIST = list( 
          'impDi'
          # 'softImpute',
          # 'mice',
          # 'imputePCA',
          # 'kNNimpute',
          # 'missForest',
          # 'Gain',
          # "GINN"
)   
         


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MISSING_RATE_LIST = c(.5,.4,.3,.2,.1) 


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
root = "../../data/randomly_missing_dataset/svmRadial_20230120"
NUM_FOLDS = 5


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
packages <- c(
  "missMDA", 
  "softImpute", 
  "caret", 
  "caTools", 
  "glue", 
  "jsonlite", 
  "future.apply", 
  "dslabs", 
  "cowplot", 
  "magick", 
  "progress", 
  "datasets", 
  "stats", 
  "foreach", 
  "fdm2id", 
  "datasetsICR", 
  "HDclassif", 
  "readxl", 
  "httr", 
  "doParallel"
)
# Install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

# Packages loading
invisible(lapply(packages, library, character.only = TRUE)) 

library(here) 
source(here('src/rscript/dimv2.R'))  
source(here('src/rscript/dpers.R'))   
source(here('src/rscript/utils.R'))    
source(here('src/rscript/imputation_comparation2.R'))     

plan(multisession, workers = 8)
library(doParallel)
registerDoParallel(cores=8) 


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#1
#DATASETS = c("iris","ionosphere", "seeds", "wine","new_thyroid", "letter", "spam")
#2
#DATASETS = c("spam")
#DATASETS = c("yeast") #"lymphography")
#DATASETS = c("iris", "new_thyroid")
#DATASETS = c("spam")
#DATASETS = c("lymphography")
#DATASETS = c("iris", "new_thyroid")
#DATASETS = c("new_thyroid", "iris", "yeast")
#DATASETS = c("spam")
# DATASETS = c("new_thyroid", "iris", "yeast")
# DATASETS = c("new_thyroid", "yeast")# "") # "new_thyroid")

#DATASETS = c("yeast", "iris", "new_thyroid")
#DATASETS = ('new_thyroid')
#DATASETS = "new_thyroid"
#DATASETS = c('iris')
#DATASETS = c("letter", "spam") #, "letter")
#DATASETS = c("iris","ionosphere", "seeds", "wine", "breast_tissue", "new_thyroid", "breast_cancer_wisconsin")# "letter", "spam")
#DATASET = c("iris")
#DATASETS = c("ionosphere", "seeds", "wine")
#DATASETS = c("iris", "ionosphere", "seeds", "wine", "breast_tissue", "new_thyroid", "breast_cancer_wisconsin")
#DATASETS = c("iris", "breast_tissue", "new_thyroid", "breast_cancer_wisconsin")
#DATASETS = c("iris", "breast_tissue", "parkinsons", "new_thyroid", "breast_cancer_wisconsin")
#DATASETS = c("iris") 


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# path = "../../data/randomly_missing_dataset/svmRadial_20230118/iris/missing_rate_50_threshold_10/1/imputed_train_impDi.csv"
# test_path = "../../data/randomly_missing_dataset/svmRadial_20230118/iris/missing_rate_50_threshold_10/1/imputed_test_impDi.csv"
# X =read.csv(path)[-1]
# Xtest = read.csv(test_path)[-1]
# 
# ypath = "../../data/randomly_missing_dataset/svmRadial_20230118/iris/missing_rate_50_threshold_10/1/y_train.csv"
# ypath_test = "../../data/randomly_missing_dataset/svmRadial_20230118/iris/missing_rate_50_threshold_10/1/y_test.csv"
# 
# y = as.vector(read.csv(ypath))$x
# #y = as.vector(read.csv(ypath))
# y = as.factor(as.numeric(factor(y)))
# y
#  
# ytest = as.vector(read.csv(ypath_test))$x
# ytest = as.factor(as.numeric(factor(ytest)))
# 
# fit.svm = train(as.data.frame(X), y, method="svmRadial")
# pred <- suppressWarnings((predict(fit.svm, as.data.frame(Xtest))))
# mean(pred ==ytest)
# pred
# ytest
# ori_path = '../../data/randomly_missing_dataset/svmRadial_20230118/breast_cancer_wisconsin/missing_rate_40_threshold_10/1/ori_X_train.csv'

# miss_path = '../../data/randomly_missing_dataset/svmRadial_20230118/breast_cancer_wisconsin/missing_rate_40_threshold_10/1/missing_X_train_normed.csv'
# X_miss = as.matrix(read.csv(miss_path))
# m_path =  '../../data/randomly_missing_dataset/svmRadial_20230118/breast_cancer_wisconsin/missing_rate_40_threshold_10/1/missing_X_train_mean.csv'
# sd_path = '../../data/randomly_missing_dataset/svmRadial_20230118/breast_cancer_wisconsin/missing_rate_40_threshold_10/1/missing_X_train_sd.csv'
# m = as.matrix(read.csv(m_path))
# sd = as.matrix(read.csv(sd_path))
# ori =  as.matrix(read.csv(ori_path))
# 
# 
# 
# rmse_calc <- function(ori_data, imputed_rescaled_data,missing_pos_filter){
#       nominator = sum((missing_pos_filter * ori_data - missing_pos_filter * imputed_rescaled_data)**2)
#       denominator = sum(missing_pos_filter) 
#       return(sqrt(nominator/denominator))
# }
# reconstructingNormedMatrix <- function(X_norm, mean, std){
#       mult = sweep(X_norm, 2, std, '*')
#       reconstrc = sweep(mult, 2, mean, '+')
#       return (reconstrc)
# } 
#   
# rmse_test = rmse_calc(
#         ori,
#         reconstructingNormedMatrix(X, m, sd),
#         is.na(X_miss)*1
#         ) 
# rmse_test



## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CalcEachFold <- function(
    fold, 
    folds, 
    dataset_name, 
    folder_name, 
    DIMV_THRESHOLD, 
    NEW_FOLDS){ 
  rmse_calc <- function(ori_data, imputed_rescaled_data,missing_pos_filter){
        nominator = sum((missing_pos_filter * ori_data - missing_pos_filter * imputed_rescaled_data)**2)
        denominator = sum(missing_pos_filter) 
        return(sqrt(nominator/denominator))
  }
  reconstructingNormedMatrix <- function(X_norm, mean, std){
        mult = sweep(X_norm, 2, std, '*')
        reconstrc = sweep(mult, 2, mean, '+')
        return (reconstrc)
  }
  get_result_path <- function(func_name){
    return (file.path(root, dataset_name, folder_name, fold, paste0("acc_rmse_", func_name ,".csv"))) 
  }
  if (TRUE){
  # if
  #      (!file.exists(get_result_path('impDi'))
  #        | ! file.exists(get_result_path('softImpute'))
  #           | ! file.exists(get_result_path('mice'))
  #              | !file.exists(get_result_path('imputePCA'))
  #                 | !file.exists(get_result_path('kNNimpute'))
  #                     | !file.exists(get_result_path('missForest'))
  #                       | !file.exists(get_result_path('Gain'))
  #                         | !file.exists(get_result_path('GINN'))
  # 
  #                   ) {
        print("start reading data")
        path.missing.X_train_normed = file.path(root, dataset_name, folder_name, fold, "missing_X_train_normed.csv")
        path.missing.X_test_normed  = file.path(root, dataset_name, folder_name, fold, "missing_X_test_normed.csv") 
        path.missing.X_train_mean   = file.path(root, dataset_name, folder_name, fold, "missing_X_train_mean.csv") 
        path.missing.X_train_sd     = file.path(root, dataset_name, folder_name, fold, "missing_X_train_sd.csv") 
        path.ori.X_train_ori        = file.path(root, dataset_name, folder_name, fold, "ori_X_train.csv")
        path.ori.X_test_ori         = file.path(root, dataset_name, folder_name, fold, "ori_X_test.csv") 
        path.y_train                = file.path(root, dataset_name, folder_name, fold, "y_train.csv")
        path.y_test                 = file.path(root, dataset_name, folder_name, fold, "y_test.csv")
         
        
        missing.X_train_normed = as.matrix(read.csv(path.missing.X_train_normed))
        missing.X_test_normed  = as.matrix(read.csv(path.missing.X_test_normed))
        missing.X_train_mean   = as.matrix(read.csv(path.missing.X_train_mean))
        missing.X_train_sd     = as.matrix(read.csv(path.missing.X_train_sd))
        ori.X_train            = as.matrix(read.csv(path.ori.X_train_ori))
        ori.X_test             = as.matrix(read.csv(path.ori.X_test_ori))
        y.train                = as.vector(read.csv(path.y_train))$x
        y.test                 = as.vector(read.csv(path.y_test))$x
        
        y.train =  as.factor(as.numeric(factor(y.train)))   
        #y.test =  as.factor(as.numeric(factor(y.test)))  
        y.test = factor(y.test, levels=levels(y.train)) 
        func_list = FUNC_LIST 
 
        
        for(j in 1:length(func_list)){
              func_name = func_list[j] 
              
              start = Sys.time()
              print(start)
              
              result_path = file.path(root, dataset_name, folder_name, fold, paste0("acc_rmse_", func_name ,".csv")) 
              # if (!file.exists(result_path)){
              if (TRUE){
                  print(paste0(folder_name, "   Start Calc for Method: ", func_name, " fold number: ", fold, "at ", start))
                  
                  path.X_train_imputed = file.path(root, dataset_name, folder_name, fold, paste0("imputed_train_", func_name, '.csv')) 
                  path.X_test_imputed  = file.path(root, dataset_name, folder_name, fold, paste0("imputed_test_", func_name, '.csv')) 
                  
                    
                      print(path.X_train_imputed)
                      print(path.X_test_imputed)
                      imputed_train = read.csv(path.X_train_imputed)[-1]
                      imputed_test  = read.csv(path.X_test_imputed)[-1]
                     
                      start = Sys.time()
                      # fit an svm to the imputed data
                      fit.svm = suppressWarnings(train(as.data.frame(imputed_train), y.train, method="svmRadial"))
                      print("--yes")
                      pred <- suppressWarnings((predict(fit.svm, as.data.frame(imputed_test))))
                      pred <- factor(pred, levels=levels(y.train))
                      print(pred)
                      print(y.test)
                      print(length(pred))
                      print(length(y.test))
                      acc = mean(pred == y.test)
                      print(acc)
                      
                      rmse_train = rmse_calc(
                              ori.X_train,
                              reconstructingNormedMatrix(imputed_train, missing.X_train_mean, missing.X_train_sd),
                              is.na(missing.X_train_normed)*1
                              )
                      rmse_test = rmse_calc(
                              ori.X_test,
                              reconstructingNormedMatrix(imputed_test, missing.X_train_mean, missing.X_train_sd),
                              is.na(missing.X_test_normed)*1
                              )
                      
                      print(rmse_train)
                      print(rmse_test)
                      print(paste0(" -------", func_name, "----", 
                                   dataset_name,"  Fold # ", fold, 
                                   " acc ", acc, " rmse_test ", rmse_test, "rmse_train ", rmse_train, 
                                   "duration ", Sys.time() - start))
                    
                      result = data.frame(
                                list("dataset" = dataset_name,
                                      "fold_number" = fold,
                                      "imputation_method" =  func_name,
                                      "accuracy" = acc,
                                      "rmse_train" = rmse_train,
                                      "rmse_test" = rmse_test
                                ) )
                      
                     write.csv(result, result_path)
                     print("done saving at ")
                     print(result_path)
              }else{
                print(paste0(">>> ", func_name, "---", dataset_name, "---",folder_name, "---", fold, "is done!!!"))
              }
              
              
        }
    }else{
      print(">>> All method in this fold are done!!! ")
    }

}
        


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CalcPipeline <- function(
    
    root, 
    DIMV_THRESHOLD, 
    MISSING_RATE, 
    NUM_FOLDS, 
    DATASETS){
  
  formatFloat2String <- function(float_value){
    i = as.integer(float_value*100) 
    s = if (as.integer(i/10) < 1){paste0("0", toString(i))}else{toString(i)} 
    return(s)
  } 
  
  
  
  folder_name = paste0(
    "missing_rate_",
    formatFloat2String(MISSING_RATE), 
    "_threshold_", 
    formatFloat2String(DIMV_THRESHOLD)
    )
  
  
  print(folder_name)
  
  count = 0
  for (dataset_name in DATASETS) {
    print(paste0("dataset ", dataset_name))

    for (i in 1:NUM_FOLDS){
        print(paste0("Start fold number: ", i, " dataset name ", dataset_name)) 
        CalcEachFold(i, folds, dataset_name, folder_name, DIMV_THRESHOLD)
    }

}
}


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

curr_dir = getwd()
if (dir.exists(file.path(root)) == F){
  dir.create(file.path(root))
}else{
  print('Root exists')
}





DIMV_THRESHOLD_LIST = c(.1)

total = length(DIMV_THRESHOLD_LIST) * length(MISSING_RATE_LIST)

print(total)
count = 0
print(DATASETS)
print(MISSING_RATE_LIST)
print("==================================")
for (DIMV_THRESHOLD in DIMV_THRESHOLD_LIST){
  for (MISSING_RATE in MISSING_RATE_LIST){

    print(">>>>>>>>>>>>>>START--------------")
    print(paste0("MISSING_RATE_", MISSING_RATE, "_THRESHOLD_", DIMV_THRESHOLD , "_NUM_FOLDS_", NUM_FOLDS))

    CalcPipeline(
        root,
        DIMV_THRESHOLD,
        MISSING_RATE,
        NUM_FOLDS,
        DATASETS)

    count = count+1
    print(paste(">>>>>>>>>>>>>>>", count, "/", total, " done"))
  }
}




## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# train_p = "../../data/randomly_missing_dataset/svmRadial_20230118/yeast/missing_rate_00_threshold_10/2/missing_X_train_normed.csv"
# test_p  = "../../data/randomly_missing_dataset/svmRadial_20230118/yeast/missing_rate_00_threshold_10/2/missing_X_test_normed.csv"
# 
# y_train_p = "../../data/randomly_missing_dataset/svmRadial_20230118/yeast/missing_rate_00_threshold_10/2/y_train.csv"
# y_test_p  = "../../data/randomly_missing_dataset/svmRadial_20230118/yeast/missing_rate_00_threshold_10/2/y_test.csv"
# 
# train =read.csv(train_p)
# test =  as.matrix(read.csv(test_p))
# 
# y.train                = as.matrix(read.csv(y_train_p))
# y_test_df                 = as.matrix(read.csv(y_test_p))
# 
# y.train =  as.factor(as.numeric(factor(y.train)))
# y.test =  factor(y_test_df, levels = levels(y.train))
# 
# 
# # fit an svm to the imputed data
# fit.svm = suppressWarnings(train(as.data.frame(train), y.train, method="svmRadial"))
# print("--yes")
# pred <- suppressWarnings((predict(fit.svm, as.data.frame(test))))
# pred <- factor(pred, levels=levels(y.train))
# print(pred)
# print(y.test)
# acc = mean(pred == y.test)
# print(acc) 


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# train_p = "../../data/randomly_missing_dataset/svmRadial_20230118/iris/missing_rate_10_threshold_10/1/missing_X_train_normed.csv"
# test_p  = "../../data/randomly_missing_dataset/svmRadial_20230118/iris/missing_rate_10_threshold_10/1/missing_X_test_normed.csv"
# train = as.matrix(read.csv(train_p)) 
# test =  as.matrix(read.csv(test_p))
# 
# train
# 
#         y.train =  as.factor(as.numeric(factor(y.train)))   
#         y.test =  as.factor(as.numeric(factor(y.test)))   

