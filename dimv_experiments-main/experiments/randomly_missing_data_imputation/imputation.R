## ----setup, include = FALSE--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(cache = TRUE, echo=TRUE, eval = TRUE)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
require(knitr)
purl("imputation_v6.Rmd", output = 'imputation_v6.R')


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#DATASETS = c("spam", "letter")
# DATASETS = c("yeast", "lymphography")

# DATASETS =c("iris","yeast", "lymphography")
# #DATASETS = c("segmentation") #c("spam")
# DATASETS =  c("new_thyroid", "yeast", "iris")
# DATASETS = c("wine", "glass")
DATASETS = c("new_thyroid", "yeast", "iris")



## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MISSING_RATE_LIST = c(.5,.4,.3,.2,.1)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

root = "../../data/randomly_missing_dataset/svmRadial_20230120"
NUM_FOLDS = 5


FUNC_LIST = list( 
    'impDi_run'
    # 'softImpute_run',
    # 'mice_run',
    # 'imputePCA_run',
    # 'kNNimpute_run'
    # 'missForest_run'
    )


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
source(here('src/rscript/dim.R'))  
source(here('src/rscript/dpers.R'))   
source(here('src/rscript/utils.R'))    
source(here('src/rscript/imputation_comparation.R'))     

plan(multisession, workers = 8)
library(doParallel)
registerDoParallel(cores=8) 


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
read_excel_url <- function(url, sheet_num){
  GET(url, write_disk(tf <- tempfile(fileext = ".xls")))
  df <- read_excel(tf, sheet=2) 
  unlink(tf)
  return(df)
}


## ---- message = FALSE, warning = FALSE---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
summary_result <- function(result, caclCol, groupByCol, fold_number, dataset_name, missing_rate, order_decreasing=TRUE){
  result$col = as.numeric(result[, caclCol]) 
  result$imputation  = result[, groupByCol]
  summary = data.frame(
                  group=levels(factor(result$imputation)), 
                  mean=(aggregate(result$col, by=list(result$imputation), FUN=mean)$x),
                  sd=(aggregate(result$col, by=list(result$imputation), FUN=sd)$x), 
                  iteration_times = max(result$fold_number)
             )
  summary = summary[order(summary$mean, decreasing=order_decreasing), ]   
  summary$dataset_name = dataset_name
  summary$missing_rate = missing_rate
  return(summary)
} 


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# train_path = "../../data/randomly_missing_dataset/svmRadial_20230118/spam/missing_rate_40_threshold_10/1/missing_X_train_normed.csv"
# test_path = "../../data/randomly_missing_dataset/svmRadial_20230118/spam/missing_rate_40_threshold_10/1/missing_X_test_normed.csv"
# 
# train = as.matrix(read.csv(train_path))
# test = as.matrix(read.csv(test_path))
# 
# 
# path.y_train                = "../../data/randomly_missing_dataset/svmRadial_20230118/spam/missing_rate_40_threshold_10/1/y_train.csv"
# path.y_test                 = "../../data/randomly_missing_dataset/svmRadial_20230118/spam/missing_rate_40_threshold_10/1/y_test.csv"
# 
# y.train                = as.matrix(read.csv(path.y_train))
# y.test                 = as.matrix(read.csv(path.y_test)) 
# 
# 
# imputed = suppressWarnings(missForest_run(train , y.train, test, y.test))  
# 
# imputed$train
# 
# imputed$test



## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
imputationEachFold <- function(
    root, 
    fold, 
    folds, 
    dataset_name, 
    folder_name, 
    DIMV_THRESHOLD
){
  get_imputed_train_path <- function(func_name){
    return(file.path(root, dataset_name, folder_name, fold, paste0("imputed_train_", func_name, '.csv')))
  }
  get_imputed_test_path <- function(func_name){
    return (file.path(root, dataset_name, folder_name, fold, paste0("imputed_test_", func_name, '.csv'))) 
  }
  get_duration <- function(func_name){
    file.path(root, dataset_name, folder_name, fold, paste0("duration_", func_name, '.csv'))
  }
    
    if (TRUE){
    # if(
      # !file.exists(get_duration('impDi'))
      #    & ! file.exists(get_duration('softImpute'))
      #       & ! file.exists(get_duration('mice'))
      #          & !file.exists(get_duration('imputePCA'))
      #             & !file.exists(get_duration('kNNimpute'))
      #                 & !file.exists(get_duration('missForest'))



      # !file.exists(get_imputed_train_path('impDi'))
      #    | ! file.exists(get_imputed_train_path('softImpute'))
      #       | ! file.exists(get_imputed_train_path('mice'))
      #          | !file.exists(get_imputed_train_path('imputePCA'))
      #             | !file.exists(get_imputed_train_path('kNNimpute'))
      #                 | !file.exists(get_imputed_train_path('missForest'))
      # 
      # 
      #   | !file.exists(get_imputed_test_path('impDi'))
      #    | ! file.exists(get_imputed_test_path('softImpute'))
      #       | ! file.exists(get_imputed_test_path('mice'))
      #          | !file.exists(get_imputed_test_path('imputePCA'))
      #             | !file.exists(get_imputed_test_path('kNNimpute'))
      #                 | !file.exists(get_imputed_test_path('missForest'))
      #               ) {

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
      y.train                = as.matrix(read.csv(path.y_train))
      y.test                 = as.matrix(read.csv(path.y_test))
          
      
    # 
      func_list = FUNC_LIST
      count = 0 
      for(j in 1:length(func_list)){
            func_name = unlist(strsplit(func_list[[j]], "_run"))[1] 
            print(paste0("- Impute fold no ", fold, " with algorithm ", func_name))
            print(dim(missing.X_train_normed))
            print(dim(missing.X_test_normed))
            
            # if (!file.exists(get_imputed_train_path(func_name)) | !file.exists(get_imputed_train_path(func_name))){
            if (TRUE){
            
            
                start = Sys.time()
                print(start)
                
                func <- get(func_list[[j]])
                
                
                print(paste0("Method: ", func_name, " fold number: ", fold))
                
                if (func_name == "impDi"){
                    
                    imputed = func(missing.X_train_normed , y.train, missing.X_test_normed, y.test, threshold=DIMV_THRESHOLD)
                }else{
                    imputed = func(missing.X_train_normed , y.train, missing.X_test_normed, y.test)  
                }
                print("complete imputation, saving result")
                
                end = Sys.time()
                
                path.X_train_imputed = file.path(root, dataset_name, folder_name, fold, paste0("imputed_train_", func_name, '.csv')) 
                path.X_test_imputed  = file.path(root, dataset_name, folder_name, fold, paste0("imputed_test_", func_name, '.csv'))
                #path.duration        = file.path(root, dataset_name, folder_name, fold, paste0("duration_", func_name, '.csv'))
                
                #duration_df = data.frame(list("start" = start, "end" = end))
                
                
                write.csv(imputed$train, path.X_train_imputed)
                write.csv(imputed$test, path.X_test_imputed)    
                #write.csv(duration, path.duration) 
                
                
                print(paste0("done saving these path, imputation duration: "))
                print(path.X_train_imputed)
                print(path.X_test_imputed)
                print(end-start)
                
                print(path.X_train_imputed)
                print(path.X_test_imputed)
            }else{
               print(paste0(">>> ", func_name, "---", dataset_name, "---",folder_name, "---", fold, "is done!!!"))
            }
          
      }
    }else{
      print("alll imputation of all method in this folds are done")
    }

}
        


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
executeImputation <- function(
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
    # dataset_conf =  get(paste0(dataset_name, "_conf")) 
    # dataset = get(dataset_name)
    # label_col = dataset_conf$label_col 
    # print(paste0("dim(dataset)=", dim(dataset)[1], dim(dataset)[2])) 
    result = NULL 
    count = 0 
    # while(is.null(result) & count<=5){
    #   try({
    #     count = count+1

  # results <- foreach::foreach(i = 1:number_of_folds, .combine='rbind') %dopar% {
  #   setTxtProgressBar(pb, i)
  #   imputeAndPredictionOnEachFold(i, missing_data, data, labels, folds, dataset_name, folder_name, DIMVthreshold)
  # }
    for (i in 1:NUM_FOLDS){
                print(paste0("Start fold number: ", i)) 
                
                # result = 
                imputationEachFold(root, i,folds, dataset_name, folder_name, DIMV_THRESHOLD)
                
      }
                
        
    #     break #break/exit the for-loop
    #   }, silent = FALSE)
    # }

}
}


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 
curr_dir = getwd()
if (dir.exists(file.path(root)) == F){
  dir.create(file.path(root))
}else{
  print('Root exists')
}
#DATASETS =c("iris","ionosphere", "seeds", "wine","new_thyroid", "letter", "spam") 
#DATASETS =c("iris", "new_thyroid") 



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

    executeImputation(
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
# root = "../../data/randomly_missing_dataset/svmRadial_20230118" 
#  dataset_name = "wine"
# folder_name = "missing_rate_10_threshold_10"
# fold = 1
#   path.missing.X_train_normed = file.path(root, dataset_name, folder_name, fold, "missing_X_train_normed.csv")
#   path.missing.X_test_normed  = file.path(root, dataset_name, folder_name, fold, "missing_X_test_normed.csv") 
#   path.missing.X_train_mean   = file.path(root, dataset_name, folder_name, fold, "missing_X_train_mean.csv") 
#   path.missing.X_train_sd     = file.path(root, dataset_name, folder_name, fold, "missing_X_train_sd.csv") 
#   path.ori.X_train_ori        = file.path(root, dataset_name, folder_name, fold, "ori_X_train.csv")
#   path.ori.X_test_ori         = file.path(root, dataset_name, folder_name, fold, "ori_X_test.csv") 
#   path.y_train                = file.path(root, dataset_name, folder_name, fold, "y_train.csv")
#   path.y_test                 = file.path(root, dataset_name, folder_name, fold, "y_test.csv")
#    
#   
#   missing.X_train_normed = as.matrix(read.csv(path.missing.X_train_normed))
#   missing.X_test_normed  = as.matrix(read.csv(path.missing.X_test_normed))
#   missing.X_train_mean   = as.matrix(read.csv(path.missing.X_train_mean))
#   missing.X_train_sd     = as.matrix(read.csv(path.missing.X_train_sd))
#   ori.X_train            = as.matrix(read.csv(path.ori.X_train_ori))
#   ori.X_test             = as.matrix(read.csv(path.ori.X_test_ori))
#   y.train                = as.matrix(read.csv(path.y_train))
#   y.test                 = as.matrix(read.csv(path.y_test))
#       
#   dim(missing.X_test_normed)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# func_list = list( 
#     'impDi_run', 
#     'softImpute_run', 
#      #'mice_run', 
#     'imputePCA_run'
#     #'kNNimpute_run'
#     # 'missForest_run'
#     )   
#   DIMVthreshold = 0.1
#   dim(missing.X_train_normed)
#   dim(y.train)
#   imputed = kNNimpute_run(missing.X_train_normed , y.train, missing.X_test_normed, y.test)
#   imputed
#   
#   # imputed = softImpute_run(missing.X_train_normed , y.train, missing.X_test_normed, y.test, threshold=DIMVthreshold) 
#   dim(imputed$train)
#   dim(imputed$test)
#   missing.X_train_normed
#   
#   missing.X_train_normed[is.na(missing.X_train_normed)] = NaN
#   S = dpers(missing.X_train_normed)
#   impDi(S, as.data.frame(missing.X_train_normed), 0.1)
#   missing.X_train_normed
#   
#   missing.X_train_normed
#   for(j in 1:length(func_list)){
#       
#         func_name = unlist(strsplit(func_list[[j]], "_run"))[1] 
#         print(paste0("- Impute fold no ", fold, " with algorithm ", func_name))
#         print(dim(missing.X_train_normed))
#         print(dim(missing.X_test_normed))
#         
#         start = Sys.time()
#         print(start)
#         
#         func <- get(func_list[[j]])
#         print(paste0("Method: ", func_name, " fold number: ", fold))
#         
#         if (func_name == "impDi"){
#             
#             imputed = func(missing.X_train_normed , y.train, missing.X_test_normed, y.test, threshold=DIMVthreshold)
#         }else{
#             imputed = suppressWarnings(func(missing.X_train_normed , y.train, missing.X_test_normed, y.test))  
#         }
#         print("complete imputation, saving result")
#         duration = Sys.time() - start 
#         path.X_train_imputed = file.path(root, dataset_name, folder_name, fold, paste0(func_name,"_train_imputed.csv.gz")) 
#         path.X_test_imputed  = file.path(root, dataset_name, folder_name, fold, paste0(func_name ,"_test_imputed.csv.gz")) 
#         
#         write.table(imputed$train, path.X_train_imputed)
#         write.table(imputed$test, path.X_test_imputed)    
#         print("done saving")
#         
#         
#   }
# } 

