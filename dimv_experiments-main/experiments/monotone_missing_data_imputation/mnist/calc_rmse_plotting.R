## ----setup, include = FALSE------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(cache = TRUE, echo=TRUE, eval = TRUE)


## --------------------------------------------------------------------------------------------------------------------------------------------
require(knitr)
FILE_NAME = 'v13'
purl("calc_rmse_plotting.Rmd", output = 'calc_rmse_plotting.R')


## --------------------------------------------------------------------------------------------------------------------------------------------

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
  "crunch"
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

plan(multisession, workers = 8)  


## --------------------------------------------------------------------------------------------------------------------------------------------
#getting the path to save 

# curr_dir = getwd()
# curr_dir
path = '../../../data/mnist/raw/'
curr_dir = getwd() 

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
  


## --------------------------------------------------------------------------------------------------------------------------------------------
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



## --------------------------------------------------------------------------------------------------------------------------------------------
# load images
processing_mnist_data <- function (){
  train = load_image_file(file.path(mnist_path, "train-images-idx3-ubyte"))
  test  = load_image_file(file.path(mnist_path,"t10k-images-idx3-ubyte")) 

  train$label =  as.factor(load_label_file(file.path(mnist_path,"train-labels-idx1-ubyte")))

  test$label = as.factor(load_label_file(file.path(mnist_path,"t10k-labels-idx1-ubyte")))
  result = list('train'=train, 'test'=test)
  return(result)
}


## ---- fig.width = 6, fig.height=1.2----------------------------------------------------------------------------------------------------------
# visualize the deleted images 
visualize_digit <- function(missing_X, y, train_removed_rows, per_col=1, per_row=8){

  par(mfcol=c(per_col, per_row))
  par(mar=c(0, 0, 3, 0), xaxs='i', yaxs='i')  
 
  for (idx in 1:(per_col*per_row)) { 
      im <- matrix(unlist(missing_X[train_removed_rows, ][idx, ]),nrow = 28 ,byrow = T)
      im <- t(apply(im, 2, rev)) 
      image(1:28, 1:28, im,  xaxt='n', main=paste(y[train_removed_rows,,drop=F][idx, ])) 
      
      #image(1:28, 1:28, im, col=gray((0:255)/255), xaxt='n', main=paste(y[train_removed_rows,,drop=F][idx, ]))
  }

}


## --------------------------------------------------------------------------------------------------------------------------------------------
processed_data = processing_mnist_data()
train = processed_data$train 
test = processed_data$test

X.train = train[, -785]
X.test = test[, -785]
y.train = train[, 785, drop=F]
y.test = test[, 785, drop=F]


## --------------------------------------------------------------------------------------------------------------------------------------------
calcRmseAndPlottingPipeline <- function(
      width_del_percent=None,
      height_del_percent=None,
      sample_deleted_percent=None,
      correlation_threshold=None,
      ori.X.train,
      ori.X.test,
      ori.y.train,
      ori.y.test, 
      ROOT
      ){

  width_del = toString(as.integer(width_del_percent*100))
  heigh_del =  toString(as.integer(height_del_percent*100)) 
  percent_img_del =  toString(as.integer(sample_deleted_percent*100)) 
  threshold =  as.integer(correlation_threshold*100)  
  
  threshold_string = if (as.integer(threshold/10) < 1){
    paste0("0", toString(threshold))
  }else{
    toString(threshold)
  } 


  sub_folder = paste0(
      "threshold_", 
      threshold_string, 
      "_deletedWidthHeightPc_", 
      width_del, 
      heigh_del, 
      '_noImagePc_', 
      percent_img_del
    ) 
  curr_dir = getwd() 
  sub_path = file.path(curr_dir, ROOT, FILE_NAME, sub_folder)
  print(sub_path)
  
  path.X_train_normed = file.path(sub_path, 'X_train_normed.csv.gz')
  path.X_test_normed  = file.path(sub_path, 'X_test_normed.csv.gz') 
  
  X_train_normed = read.table(path.X_train_normed, sep=' ', header=TRUE)  
  X_test_normed  = read.table(path.X_test_normed, sep=' ', header=TRUE)  
  
  print(dim(X_train_normed))
  print(dim(X_test_normed))
  
  test_removed_rows = as.vector(read.csv(file.path(sub_path, 'test_removed_rows.csv')))$test_removed_rows
  
  get_normalization_parameters <- function(Xtrain){
      mean = apply(Xtrain, 2, mean, na.rm=TRUE)
      sd = apply(Xtrain, 2, sd, na.rm=TRUE) 
      result = list("mean" = mean, "sd" = sd)
  } 
  
  train_missing = ori.X.train
  train_missing_pos_filter = is.na(X_train_normed) 
  train_missing[train_missing_pos_filter] <- NA  
  
  test_missing = ori.X.test
  test_missing_pos_filter = is.na(X_test_normed)
  test_missing[test_missing_pos_filter] <- NA  
  
  params     = get_normalization_parameters(train_missing)
  train_mean = params$mean
  train_sd   = params$sd 
  
  algo_list = c('impDi', 'softImpute', 'Gain')
  
  
  plot_path = file.path(paste0("../../../data/mnist/plot/", sub_folder)) 
  if (dir.exists(plot_path)==F){
    dir.create(plot_path)
  }  
  c = 0
  for (algo in algo_list){
        path.train_imputed = file.path(sub_path, paste0('train_', algo, '.csv.gz'))
        path.test_imputed  = file.path(sub_path, paste0('test_', algo, '.csv.gz')) 
        
        train_imputed     = read.table(path.train_imputed, sep=' ', header=TRUE)  
        test_imputed      = read.table(path.test_imputed, sep=' ', header=TRUE)  
        
        train_rescaled    = reconstructingNormedMatrix(train_imputed, train_mean, train_sd)
        test_rescaled     = reconstructingNormedMatrix(test_imputed, train_mean, train_sd)
        
        
        rmse_train       = calc_rmse(ori.X.train, train_rescaled, train_missing_pos_filter*1)
        rmse_test        = calc_rmse(ori.X.test, test_rescaled, test_missing_pos_filter*1)
        
        result = data.frame(list("name" = sub_folder, "method" = algo, "rmse_train" = rmse_train, "rmse_test" = rmse_test))
        
        
        
        if (c ==0 ){
          results = result
        }else{
          results = rbind(results, result)
        }
        c = c + 1
        
        plotRescaledImg(
              plot_path, 
              sub_folder,
              train_rescaled, 
              algo, 
              ori.X.test, 
              test_removed_rows, 
        )
  }
  
  # SAVE RESULT
  write.csv(results, file.path(data_dir, 'rmse.csv')) 
  
  OriImgPath <- file.path(plot_path, "ori_test.png")  
  png(OriImgPath, width = dev.size("px")[1], height = dev.size("px")[2]) 
  visualize_digit(ori.X.test, ori.y.test, test_removed_rows, 1, 8) 
  dev.off()
  
  OriMissingImgPath <- file.path(plot_path, "ori_missing_test.png")  
  png(OriMissingImgPath, width = dev.size("px")[1], height = dev.size("px")[2]) 
  visualize_digit(test_missing_pos_filter, ori.y.test, test_removed_rows, 1, 8) 
  dev.off() 
} 


## ---- fig.width = 6, fig.height=1.2----------------------------------------------------------------------------------------------------------
plotRescaledImg <- function(
    plot_path, 
    sub_folder, 
    Xtest_recon, 
    algo, 
    y_test, 
    test_removed_rows
  ){ 
  
  if (dir.exists(plot_path)==F){
    dir.create(plot_path)
  } 
    
  ImgPath <- file.path(plot_path, paste0(algo, "_test.png"))
  png(ImgPath, width=dev.size("px")[1] , height = dev.size("px")[2])  
  visualize_digit(Xtest_recon, y_test, test_removed_rows, 1, 8) 
  dev.off()
}


## --------------------------------------------------------------------------------------------------------------------------------------------
width_height_percentages =c(.6, .5, .4)
sample_deleted_percentages = c(.5)
correlation_threshold =c(.1)
ROOT = '../../../data/mnist/imputed' 

for (width_height_pc in width_height_percentages){
  for (sample_pc in sample_deleted_percentages){
    for (th in correlation_threshold){
      calcRmseAndPlottingPipeline(
                  width_del_percent=width_height_pc,
                  height_del_percent=width_height_pc,
                  sample_deleted_percent=sample_pc,
                  correlation_threshold=correlation_threshold,
                  X.train,
                  X.test,
                  y.train,
                  y.test, 
                  ROOT
                  )
    }

  }

}

 


## --------------------------------------------------------------------------------------------------------------------------------------------
# write.table(impDi_Xtest_recon[1:88, 1:88],path) 


## --------------------------------------------------------------------------------------------------------------------------------------------
# path = "../../../data/mnist/imputed/v13/threshold_30_deletedWidthHeightPc_5050_noImagePc_50/test_Gain.csv.gz"
# d = read.table(path, sep=' ', header=TRUE)
# d



## --------------------------------------------------------------------------------------------------------------------------------------------
# path01= file.path(paste0("../../../data/mnist/plot/", "file_test.txt") )
# path = "../../../data/mnist/imputed/v13/threshold_30_deletedWidthHeightPc_5050_noImagePc_50/X_train_normed.csv.gz"
# #f = read.table(gzfile(path)))
# myData <- read.table(path, sep=' ', header = TRUE)
# myData



## ---- fig.width = 6, fig.height=1.2----------------------------------------------------------------------------------------------------------
# 
# OriImgPath <- file.path(plot_path, "ori_test.png")  
# png(OriImgPath, width = dev.size("px")[1], height = dev.size("px")[2]) 
# visualize_digit(test_ori, y_test, test_removed_rows, 1, 8) 
# dev.off()
# 
# OriMissingImgPath <- file.path(plot_path, "ori_missing_test.png")  
# png(OriMissingImgPath, width = dev.size("px")[1], height = dev.size("px")[2]) 
# visualize_digit(test_ori_missing, y_test, test_removed_rows, 1, 8) 
# dev.off() 
# 
#   plot_path = file.path(paste0("../../../data/mnist/plot/", sub_folder))
#   if (dir.exists(plot_path)==F){
#     dir.create(plot_path)
#   }   
#   softImputeImgPath <- file.path(plot_path, "softImpute_test.png") 
#   png(softImputeImgPath, width=dev.size("px")[1] , height = dev.size("px")[2])  
#   visualize_digit(softImpute_Xtest_recon, y_test, test_removed_rows, 1, 8) 
#   dev.off()
#   
#   GainImgPath <- file.path(plot_path, "Gain_test.png")  
#   png(GainImgPath, width = dev.size("px")[1], height = dev.size("px")[2]) 
#   visualize_digit(GAIN_Xtest_recon, y_test, test_removed_rows, 1, 8) 
#   dev.off() 
#    
#   impDiImgPath <- file.path(plot_path, "impDi_test.png")  
#   png(impDiImgPath, width = dev.size("px")[1], height = dev.size("px")[2]) 
#   visualize_digit(impDi_Xtest_recon, y_test, test_removed_rows, 1, 8) 
#   dev.off()
  
  
   
#   merged_width = 300 #dev.size("px")[1]
#   merged_height = 300*5+50 #dev.size("px")[2]*2+50
#   
#   p1 <- ggdraw() + draw_image(OriImgPath)
#   p2 <- ggdraw() + draw_image(OriMissingImgPath)
#   p3 <- ggdraw() + draw_image(softImputeImgPath)
#   p4 <- ggdraw() + draw_image(GainImgPath) 
#   p5 <- ggdraw() + draw_image(impDiImgPath)  
#     
#   imgPath =  file.path(plot_path, "img.png")  
#   
#   png(imgPath) #,  width = merged_width, height = merged_height) 
#   
#   plot_grid(p1, p2, p3, p4, p5, 
#             nrow = 5, 
#             ncol=1,
#             labels =  NULL, 
#             label_size = 12, 
#             scale=1,
#             vjust=c(2.0, 2.0, 2.0, 2.0),
#             label_colour = "blue")
#   
#   dev.off()
#   # 
#   print(paste0("Complete saving plot, pipeline is done ", sub_folder))  
# }

