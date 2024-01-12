## ----setup, include = FALSE---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(cache = TRUE, echo=TRUE, eval = TRUE)


## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
require(knitr)
FILE_NAME = 'v13'
purl("imputation.Rmd", output = 'imputation.R')


## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
  "progress"
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

plan(multisession, workers = 4)  



## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#getting the path to save 

curr_dir = getwd()
path = '../../../data/mnist/raw/'

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
  


## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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



## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# load images
processing_mnist_data <- function (){
  train = load_image_file(file.path(mnist_path, "train-images-idx3-ubyte"))
  test  = load_image_file(file.path(mnist_path,"t10k-images-idx3-ubyte")) 

  train$label =  as.factor(load_label_file(file.path(mnist_path,"train-labels-idx1-ubyte")))

  test$label = as.factor(load_label_file(file.path(mnist_path,"t10k-labels-idx1-ubyte")))
  result = list('train'=train, 'test'=test)
  return(result)
}


## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
processed_data = processing_mnist_data()
train = processed_data$train 
test = processed_data$test
X.train = train[, -785]
X.test = test[, -785]
y.train = train[, 785, drop=F]
y.test = test[, 785, drop=F] 


## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
get_image_position_spatial_to_flatten<- function(delImgPosWidth, delImgPosHeight){ 
  # delImgPosHeight: row 
  # delImgPosWeight : col 
  tmp = c(1:784)
  im <- matrix(unlist(tmp),nrow = 28,byrow = T)
  idxs = im[delImgPosHeight, delImgPosWidth]  
  return(matrix(idxs,nrow = 1,byrow = T)[, ])
}


## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
image_edge_deleting <- function(
    data, 
    delete_type, #by_percent, by_pixel_number 
    percents_of_data,
    image_width,
    image_height,
    width_del_percent=0, 
    height_del_percent=0,
    from_pixel_width=None, 
    from_pixel_height=None
    ){
  if (delete_type =='by_percent'){
    n = dim(data)[2]
    from_pixel_width = ceiling((1-width_del_percent)*image_width)   
    from_pixel_height = ceiling((1-height_del_percent)*image_height)
  }
  if (delete_type=='by_pixel_number'){
    from_pixel_width = from_pixel_width
    from_pixel_height = from_pixel_height
  }
  
flatten_columns_removed = get_image_position_spatial_to_flatten(
  from_pixel_width:image_width,
  from_pixel_height: image_height
)

  flatten_rows_removed = sample.int(nrow(data), as.integer(nrow(data)*percents_of_data))
  missing_data = data
  missing_data[flatten_rows_removed, flatten_columns_removed] <- NA
  result = list(
    'missing_data'=missing_data, 
    'flatten_columns_removed'=flatten_columns_removed,  
    'flatten_rows_removed'=flatten_rows_removed
  ) 
  return(result)
}



## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# visualize the deleted images 
visualize_digit <- function(missing_X, y, train_removed_rows, per_col, per_row, title){

  par(mfcol=c(per_col, per_row))
  par(mar=c(0, 0, 3, 0), xaxs='i', yaxs='i')  
 
  for (idx in 1:(per_col*per_row)) { 
      im <- matrix(unlist(missing_X[train_removed_rows, ][idx, ]),nrow = 28,byrow = T)
      im <- t(apply(im, 2, rev)) 
      image(1:28, 1:28, im,  xaxt='n', main=paste(y[train_removed_rows,,drop=F][idx, ])) 
      #image(1:28, 1:28, im, col=gray((0:255)/255), xaxt='n', main=paste(y[train_removed_rows,,drop=F][idx, ]))
  }

} 



## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

sampling_data <- function(data, y_col_name, sample_perc){
  data$label= data[, y_col_name]
  sample_indices <- sample.split(Y=data[, 'label'], SplitRatio = sample_perc)
  sample_data <- as.data.frame(subset(data, sample_indices == TRUE))
  result = list(
    'sample'=sample_data, 
    'X'=as.matrix(sample_data[, 1:(28*28)]), 
    'y'=sample_data[,'label', drop=F]
  )
  return(result)
}



## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
normalizing <- function(x=None, Xtrain=None){
  na_mask = is.na(x)
  mean = apply(Xtrain, 2, mean, na.rm=TRUE)
  sd = apply(Xtrain, 2, sd, na.rm=TRUE)
  
  sd_equal_zero_mask = which(sd==0)
  subtract_mean = sweep(x, 2, mean, '-')
  X_normed = sweep(subtract_mean, 2, sd, "/")
  
  X_normed[is.na(X_normed)] = 0 
  X_normed[is.infinite(X_normed)] = 0 
  X_normed[na_mask] = NA 
  result = list('X_normed'=X_normed, 'mean'=mean, 'sd'=sd, 'sd_equal_zero_mask'=sd_equal_zero_mask)
  return (result) 
}

reconstructingNormedMatrix <- function(X_norm, mean, std){
  mult = sweep(X_norm, 2, std, '*')
  reconstrc = sweep(mult, 2, mean, '+')
  return (reconstrc)
} 


## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
mnistDataPreparation <- function(
    width_del_percent=None, 
    height_del_percent=None, 
    sample_deleted_percent=None, 
    X.train, 
    X.test, 
    y.train, 
    y.test
    ){
  X_train = as.matrix(X.train)
  X_test = as.matrix(X.test)
  y_train = as.matrix(y.train)
  y_test = as.matrix(y.test) 
  
  #cut a piece of image 
  removed_train = image_edge_deleting(
    X_train,
    'by_percent', 
    sample_deleted_percent, 28, 28, 
    width_del_percent=width_del_percent,
    height_del_percent=height_del_percent)
  
  removed_test= image_edge_deleting(
    X_test,
    'by_percent', 
    sample_deleted_percent, 28,28,
    width_del_percent=width_del_percent,
    height_del_percent=height_del_percent)  
  
  train_removed_rows = removed_train$flatten_rows_removed
  test_removed_rows = removed_test$flatten_rows_removed
  train_removed_columns = removed_train$flatten_columns_removed 
  test_removed_columns = removed_test$flatten_columns_removed 
  
  missing.X_train = removed_train$missing_data
  missing.X_test =  removed_test$missing_data  
  # normalization 
  train_normed = normalizing(x=missing.X_train,Xtrain=missing.X_train)
  missing.X_train_normed = train_normed$X_normed
  missing.X_train_mean = train_normed$mean
  missing.X_train_sd = train_normed$sd 
  
  test_normed = normalizing(x=missing.X_test, Xtrain=missing.X_train)
  missing.X_test_normed = test_normed$X_normed
  
  result = list(
    "missing.X_train_normed" = missing.X_train_normed,
    "y_train" = y_train, 
    "missing.X_test_normed" = missing.X_test_normed, 
    "y_test" = y_test, 
    "train_removed_rows" = train_removed_rows, 
    "test_removed_rows" = test_removed_rows, 
    "missing.X_train_mean" = missing.X_train_mean, 
    "missing.X_train_sd" = missing.X_train_sd 
  )
  return(result) 
}



## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
imputationPipeline<- function(
    width_del_percent=None,
    height_del_percent=None,
    sample_deleted_percent=None,
    correlation_threshold=None,
    X.train,
    X.test,
    y.train,
    y.test
){
  width_del_percent = width_del_percent
  height_del_percent = height_del_percent
  sample_deleted_percent = sample_deleted_percent
  X.train = X.train 
  X.test = X.test
  y.train = y.train 
  y.test = y.test 
  
  missingDataCreated = mnistDataPreparation( 
      width_del_percent=width_del_percent, 
      height_del_percent=height_del_percent, 
      sample_deleted_percent=sample_deleted_percent, 
      X.train, 
      X.test, 
      y.train, 
      y.test
  )
  print(paste0(
    "Starting imputation with weight and weight pc : ", 
    width_del_percent, 
    ",  sample deleted percent: ", 
    sample_deleted_percent, 
    ", correlation threshold: ", 
    correlation_threshold
    )) 
  
  missing.X_train_normed=missingDataCreated$missing.X_train_normed 
  y_train = missingDataCreated$y_train
  
  missing.X_test_normed = missingDataCreated$missing.X_test_normed
  y_test = missingDataCreated$y_test
  
  train_removed_rows = missingDataCreated$train_removed_rows
  test_removed_rows = missingDataCreated$test_removed_rows
  
  missing.X_train_mean = missingDataCreated$missing.X_train_mean
  missing.X_train_sd = missingDataCreated$missing.X_train_sd
  #---------------------------------------------------------------------------------------------------------------  
  START = Sys.time()   
  result_softImpute = softImpute_run(
    missing.X_train_normed, 
    y_train,  
    missing.X_test_normed, 
    y_test
  ) 
   
  softImputeTime = Sys.time() - START 
  print(softImputeTime)
  
  
  softImpute.Xrecon.test = reconstructingNormedMatrix(
    result_softImpute$test, 
    missing.X_train_mean, 
    missing.X_train_sd 
    )  
  
  softImpute.Xrecon.train = reconstructingNormedMatrix(
    result_softImpute$train, 
    missing.X_train_mean, 
    missing.X_train_sd 
  )   
  #---------------------------------------------------------------------------------------------------------------  
  
  START = Sys.time()   
  result_impDi = impDi_run(
    missing.X_train_normed, 
    y_train,  
    missing.X_test_normed, 
    y_test, 
    correlation_threshold  
  ) 
  impDiTime = Sys.time() - START  
  print(impDiTime)
  
  impDi.Xrecon.test = round(reconstructingNormedMatrix(
    result_impDi$test, 
    missing.X_train_mean , 
    missing.X_train_sd 
    ))
  
  impDi.Xrecon.train = round(reconstructingNormedMatrix(
    result_impDi$train, 
    missing.X_train_mean , 
    missing.X_train_sd
    )) 
  #------------ 
  curr_dir = getwd() 
  main_dir = paste0('../../../data/mnist/imputed/', FILE_NAME)
      
  width_del = toString(as.integer(width_del_percent*100))
  heigh_del =  toString(as.integer(height_del_percent*100)) 
  percent_img_del =  toString(as.integer(sample_deleted_percent*100)) 
  threshold =  as.integer(correlation_threshold*100) 
  
  threshold_string = if (as.integer(threshold/10) < 1){paste0("0", toString(threshold))}else{toString(threshold)} 
  
  sub_folder = paste0(
    "threshold_", 
    threshold_string, 
    "_deletedWidthHeightPc_", 
    width_del, 
    heigh_del, 
    '_noImagePc_', 
    percent_img_del
  )
  
  sub_path = file.path(curr_dir, main_dir, sub_folder)
  if (dir.exists(sub_path)==F){
    dir.create(sub_path)
  }
  print(paste0("Imputation is done, start saving result at ", sub_path))
  # writing result 
  write.table(missing.X_train_normed, file.path(sub_path, "X_train_normed.csv.gz"), row.names=FALSE)
  write.table(missing.X_test_normed, file.path(sub_path, "X_test_normed.csv.gz"), row.names=FALSE)
  
  # imputed data
  write.table(result_impDi$train, file.path(sub_path, 'train_impDi.csv.gz'), row.names=FALSE)
  write.table(result_impDi$test, file.path(sub_path, 'test_impDi.csv.gz'), row.names=FALSE)
  write.table(result_softImpute$train, file.path(sub_path, 'train_softImpute.csv.gz'), row.names=FALSE)
  write.table(result_softImpute$test, file.path(sub_path, 'test_softImpute.csv.gz'), row.names=FALSE)
  
  #deleted rows
  write.csv(as.data.frame(train_removed_rows), file.path(sub_path, 'train_removed_rows.csv'), row.names=FALSE)
  write.csv(as.data.frame(test_removed_rows), file.path(sub_path, 'test_removed_rows.csv'), row.names=FALSE)
  
  # #rescaled as original size of image
  # write.table(impDi.Xrecon.train, file.path(sub_path, 'train_impDi_Xrecon.csv.gz'), row.names=FALSE)
  # write.table(impDi.Xrecon.test, file.path(sub_path, 'test_impDi_Xrecon.csv.gz'), row.names=FALSE)
  # write.table(softImpute.Xrecon.train, file.path(sub_path, 'train_softImpute_Xrecon.csv.gz'), row.names=FALSE)
  # write.table(softImpute.Xrecon.test, file.path(sub_path, 'test_softImpute_Xrecon.csv.gz'), row.names=FALSE)

  # #labeled
  # write.csv(y_train, file.path(sub_path, 'y_train.csv'), row.names=FALSE)
  # write.csv(y_test, file.path(sub_path, 'y_test.csv'), row.names=FALSE)
  
  # imputation_time <- vector(mode='list', length = 2)
  # imputation_time[[1]]<c('softImpute', 'DIMV')
  # imputation_time[[2]]<c(softImputeTime, impDiTime)
  # exportJson <- toJSON(imputation_time)
  # write(exportJson, 'imputation_time.json') 
  
  #saving plot ---------------
  # print("Done saving result, Start saving plot")
  # softImputeImgPath <- file.path(sub_path, "softImpute_test.png") 
  # png(softImputeImgPath, width=dev.size("px")[1] , height = dev.size("px")[2])  
  # visualize_digit(softImpute.Xrecon.test, y_test, test_removed_rows, 2, 6) 
  # dev.off()
  # 
  # impDiImgPath <- file.path(sub_path, "impDi_test.png")  
  # png(impDiImgPath, width = dev.size("px")[1], height = dev.size("px")[2]) 
  # visualize_digit(impDi.Xrecon.test, y_test, test_removed_rows, 2, 6) 
  # dev.off()
  # 

}


## -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 # width_height_percentages =c(.6)
 # sample_deleted_percentages = c(.5)
 # correlation_threshold =c(.05, .1, .2, .3, .4, .5,.6,.7)

 width_height_percentages =c(.4)
 sample_deleted_percentages = c(.5)
 correlation_threshold =c(.1)

# width_height_percentages =c(.4) 
# sample_deleted_percentages = c(.5)
# correlation_threshold = c(.05) 
for (width_height_pc in width_height_percentages){
  for (sample_pc in sample_deleted_percentages){
    for (th in correlation_threshold){
      imputationPipeline(
        width_del_percent=width_height_pc,
        height_del_percent=width_height_pc,
        sample_deleted_percent=sample_pc,
        correlation_threshold=th,
        X.train,
        X.test,
        y.train,
        y.test
        )
    }

  }

}



