## ----setup, include = FALSE---------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(cache = TRUE, echo=TRUE, eval = TRUE)


## -----------------------------------------------------------------------------------------------------------------------------------------------
require(knitr)
purl("imputation_mnist.Rmd", output = 'imputation_mnist.R')


## -----------------------------------------------------------------------------------------------------------------------------------------------
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
source(here('src/rscript/dimv.R'))  
source(here('src/rscript/dpers.R'))   
source(here('src/rscript/utils.R'))    
source(here('src/rscript/imputation_comparation.R'))     

plan(multisession, workers = 5)
library(doParallel)


## -----------------------------------------------------------------------------------------------------------------------------------------------
#getting the path to save 

curr_dir = getwd()
curr_dir
path = '../../data/mnist/raw/'

mnist_path = file.path(curr_dir, path) 
print(mnist_path)

if (!file.exists(file.path(mnist_path, "train-images-idx3-ubyte")) |
  !file.exists(file.path(mnist_path, "t10k-images-idx3-ubyte")) |
  !file.exists(file.path(mnist_path, "train-labels-idx1-ubyte")) |
  !file.exists(file.path(mnist_path, "t10k-labels-idx1-ubyte")) 
  ){
  
  # getting the data 
  mnist <- read_mnist(
    path = NULL,
    destdir = mnist_path, 
    download = TRUE,
    url = "https://www2.harvardx.harvard.edu/courses/IDS_08_v2_03/",
    keep.files = TRUE
  )  
  
  # clear folder data (avoid wrong zipping)
  list_files = list.files(path=mnist_path) 
  for (x in 1:length(list_files)){
    file_path = file.path(mnist_path, list_files[x]) 
    if (substring(file_path, nchar(file_path)-2, nchar(file_path)) == '.gz'){
      R.utils::gunzip(file_path, overwrite=TRUE, remove=FALSE) 
    }
    }  
} 
  


## -----------------------------------------------------------------------------------------------------------------------------------------------
# load image files
load_image_file = function(filename) {
  ret = list()
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n    = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  nrow = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  ncol = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  x = readBin(f, 'integer', n = n * nrow * ncol, size = 1, signed = FALSE)
  close(f)
  data.frame(matrix(x, ncol = nrow * ncol, byrow = TRUE))
}

# load label files
load_label_file = function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  y
} 



## -----------------------------------------------------------------------------------------------------------------------------------------------
# load images
processing_mnist_data <- function (){
  train = load_image_file(file.path(mnist_path, "train-images-idx3-ubyte"))
  test  = load_image_file(file.path(mnist_path,"t10k-images-idx3-ubyte")) 

  train$label =  as.factor(load_label_file(file.path(mnist_path,"train-labels-idx1-ubyte")))

  test$label = as.factor(load_label_file(file.path(mnist_path,"t10k-labels-idx1-ubyte")))
  result = list('train'=train, 'test'=test)
  return(result)
}


## -----------------------------------------------------------------------------------------------------------------------------------------------
processed_data = processing_mnist_data()
train = processed_data$train 
test = processed_data$test

X.train = train[, -785]
X.test = test[, -785]
y.train = train[, 785, drop=F]
y.test = test[, 785, drop=F]

# 





# train_idx = sample(1:nrow(X.train))
# test_idx = sample(1:nrow(X.test))
# #
# 
# X.train = train[, -785][train_idx, ,drop=F][1:10000,, drop=F]
# X.test = test[, -785][test_idx, ,drop=F][1:1000,,drop=F]
# 
# y.train = train[, 785, drop=F][train_idx,,drop=F][1:10000,,drop=F]
# y.test = test[, 785, drop=F][test_idx,,drop=F ][1:1000,,drop=F]
# 
# 
# X.train = train[, -785][train_idx, ,drop=F][1:1000, 550:555, drop=F]
# X.test = test[, -785][test_idx, ,drop=F][1:100,550:555,drop=F]
# 
# y.train = train[, 785, drop=F][train_idx,,drop=F][1:1000,,drop=F]
# y.test = test[, 785, drop=F][test_idx,,drop=F ][1:100,,drop=F]


## -----------------------------------------------------------------------------------------------------------------------------------------------
createRandomlyMissingData = function(data, rate){
  data = as.matrix(data)
  col_num = dim(data)[2] 
  flatten = as.vector(data) 
  
    
  mask = runif(length(flatten), min = 0, max = 1) < rate
  flatten[mask]=NaN
  return(matrix(flatten, ncol = col_num))
}


## -----------------------------------------------------------------------------------------------------------------------------------------------
mnistDataPreparation <- function(
    missing_rate=None, 
    X.train, 
    X.test, 
    y.train, 
    y.test
    ){
  
  X_train = as.matrix(X.train)
  X_test = as.matrix(X.test)
  
  missing.X_train = createRandomlyMissingData(X.train, missing_rate) 
  missing.X_test =  createRandomlyMissingData(X.test, missing_rate) 
  
  # normalization 
  train_normed = normalizing(x=missing.X_train,Xtrain=missing.X_train)
  missing.X_train_normed = train_normed$X_normed
  missing.X_train_mean = train_normed$mean
  missing.X_train_sd = train_normed$sd 
  
  test_normed = normalizing(x=missing.X_test, Xtrain=missing.X_train)
  missing.X_test_normed = test_normed$X_normed

  
  result = list(
    "missing.X_train_normed" = missing.X_train_normed,
    "missing.X_test_normed" = missing.X_test_normed, 
    "missing.X_train_mean" = missing.X_train_mean, 
    "missing.X_train_sd" = missing.X_train_sd 
  )
  return(result) 
}
 


## -----------------------------------------------------------------------------------------------------------------------------------------------

imputationPipeline <- function(path, DIMV_THRESHOLD, MISSING_RATE, X.train, y.train, X.test, y.test, dataset_name){
  print("start")
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
  
  X.train = X.train 
  X.test = X.test
  y.train = y.train 
  y.test = y.test 
  
  missingDataCreated = mnistDataPreparation( 
      missing_rate=MISSING_RATE, 
      X.train, 
      X.test, 
      y.train, 
      y.test
  )
  
  missing.X_train_normed=missingDataCreated$missing.X_train_normed 
  missing.X_test_normed = missingDataCreated$missing.X_test_normed
  missing.X_train_mean = missingDataCreated$missing.X_train_mean
  missing.X_train_sd = missingDataCreated$missing.X_train_sd 
  print(which(rowSums(is.na(missing.X_train_normed))==50))
  
  
  #saving data 
  sub_path = file.path(path, folder_name) 
  if (dir.exists(file.path(sub_path)) == F){ dir.create(file.path(sub_path))}else{print('Root exists')}
  path.missing.X_train_normed = file.path(sub_path,  "missing_X_train_normed.csv.gz")
  path.missing.X_test_normed = file.path(sub_path,  "missing_X_test_normed.csv.gz") 
  path.missing.X_train_mean =  file.path(sub_path,  "missing_X_train_mean.csv.gz") 
  path.missing.X_train_sd = file.path(sub_path,   "missing_X_train_sd.csv.gz")
  
  write.table(missing.X_train_normed, path.missing.X_train_normed)
  write.table(missing.X_test_normed, path.missing.X_test_normed)
  write.table(missing.X_train_mean, path.missing.X_train_mean)
  write.table(missing.X_train_sd, path.missing.X_train_sd) 
  
  
  func_list = list( 
      'impDi_run', 
      'softImpute_run'
      )   
  
  for(j in 1:length(func_list)){
        start = Sys.time()
        print(start)
        func_name = unlist(strsplit(func_list[[j]], "_run"))[1]

        func <- get(func_list[[j]])
        if (func_name == "impDi"){
          impted = func(missing.X_train_normed , y.train, missing.X_test_normed, y.test, threshold=DIMV_THRESHOLD, print_progression = T, workers=7, run_parallel = T)
        }else{
          impted = suppressWarnings(func(missing.X_train_normed , y.train, missing.X_test_normed, y.test))
        }
        duration = (Sys.time() - start)/60
        print(paste0("Imputation is done, method ", func_name, "minutes ", duration))

        print(paste0("dimention of train: ", dim(impted$train)))
        print(paste0("dimention of test: ", dim(impted$test)))


        # saving imputed data
        write.table(impted$train, file.path(sub_path, paste0('train', func_name, '.csv.gz')), row.names=FALSE)
        write.table(impted$test, file.path(sub_path, paste0('test', func_name, '.csv.gz')), row.names=FALSE)
        print(paste0("Sving imputed is done, method ", func_name, "minutes ", duration))
  }
}


## -----------------------------------------------------------------------------------------------------------------------------------------------
dataset_name = 'mnist' 
root = file.path("../../data/randomly_missing_dataset", dataset_name) 
version = 'v1' 
path  = file.path(root,version)
path


curr_dir = getwd()
if (dir.exists(file.path(root)) == F){ dir.create(file.path(root))}else{print('Root exists')}
if (dir.exists(file.path(path)) == F){ dir.create(file.path(path))}else{print('Path exists')}


DIMV_THRESHOLD_LIST = c(.6)
MISSING_RATE_LIST = c(.2,.3,.4,.5) 
total = length(DIMV_THRESHOLD_LIST) * length(MISSING_RATE_LIST)
count = 0 

print("==================================")
for (DIMV_THRESHOLD in DIMV_THRESHOLD_LIST){
  for (MISSING_RATE in MISSING_RATE_LIST){
    print(paste0(">>>>>>>>>>>>>>> ", count, " / ", total, " done"))
    print(">>>>>>>>>>>>>>START--------------")
    print(paste0("MISSING_RATE_", MISSING_RATE, "_THRESHOLD_", DIMV_THRESHOLD ))

    imputationPipeline(path, DIMV_THRESHOLD, MISSING_RATE, X.train, y.train, X.test, y.test, dataset_name) 

    count = count+1
  }
}





## -----------------------------------------------------------------------------------------------------------------------------------------------


