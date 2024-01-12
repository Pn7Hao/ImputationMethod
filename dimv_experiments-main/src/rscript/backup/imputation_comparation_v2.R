packages <- c(
  "missMDA", 
  "scales", 
  "future.apply", 
  "scales",
  "caTools", 
  "softImpute", 
  "mice", 
  "missForest", 
  "caret", 
  "future"
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



impDi_run <- function(X.train, y.train, X.test, y.test, threshold=.3){ 
  #a) on training set 
  X.train[is.na(X.train)] <- NaN
  sigmaDper = dpers(X.train)
  
  X_imp.train = impDi(sigmaDper, X.train, threshold)[,, drop=F]
  #b) on testing set   
  X.test[is.na(X.test)] <- NaN 
  X.combine = rbind(X.test, X.train)   
  
  X_imp.test = impDi(sigmaDper, X.combine, threshold)[,, drop=F][1: nrow(X.test),,drop=F]
  
  result = list("train" = X_imp.train, "test" = X_imp.test)
  return(result)
} 

softImpute_run <- function(X.train, y.train, X.test, y.test){
  #a) on training set
  fit_train = softImpute(as.matrix(X.train) , type = 'als') 
  X_imp.train = softImpute::complete(
                              as.matrix(X.train), 
                              fit_train)[,,drop=F]
  #b) on testing set 
  X.test[is.na(X.test)] <- NaN 
  X.combine = rbind(X.test, X.train) 
  
  fit_test = softImpute(as.matrix(X.combine) , type = 'als') 
  X_imp.test = softImpute::complete(
                            as.matrix(X.combine), 
                            fit_test)[1: nrow(X.test),,drop=F]
  result = list("train" = X_imp.train, "test" = X_imp.test )
  return(result)
}  

mice_run <- function(X.train, y.train, X.test, y.test){
  #a) on training set 
  # parameter to set here 
  X_mice.train = mice(X.train, print=FALSE)
  X_imp.train = as.matrix(mice::complete(X_mice.train))
  
  #b) on testing set 
  X.combine = rbind(X.test, X.train)  
  imp.combine <-mice(X.combine, 
                        ignore = c(
                           rep(TRUE, dim(X.test)[1]), 
                           rep(FALSE, dim(X.train)[1])), 
                     print=FALSE)
  X_imp.test = as.matrix(complete(imp.combine)[1: nrow(X.test),,drop=F]) 
  result = list("train" = X_imp.train, "test" = X_imp.test)
  return(result)
} 

imputePCA_run <-function(X.train, y.train, X.test, y.test){
  #a) on training set 
  ncomp = estim_ncpPCA(X.train, ncp.min=2)
  fit_train= imputePCA(X.train, ncp= ncomp$ncp)
  X_imp.train = fit_train$completeObs[,,drop=F]
  pca <- prcomp(fit_train$completeObs, scale.=T, rank. = ncomp$ncp)  
  
  #b) on testing set 
  missing <- is.na(X.test) 
  X_imp.test = X.test
  X_imp.test[missing] = 0 #initialize missing with 0  
  test_pca <- predict(pca , newdata = X_imp.test)
  test_pca_reconstructed <-  t(t(test_pca %*% t(pca$rotation)) * pca$scale + pca$center)   
  X_imp.test[missing]=test_pca_reconstructed[missing]
  result = list("train" = X_imp.train, "test" = X_imp.test[,,drop=F ])
  return(result)
}  

kNNimpute_run <-function(X.train, y.train, X.test, y.test){
  #a) on training set 
  X_imp.train = bnstruct::knn.impute(X.train, k=5)   
  
  #b) on testing set 
  X.test_missing = as.matrix(X.test[rowSums(is.na(X.test)) > 0 ,,drop=F]) 
  X.combine = rbind(X.test_missing, X.train) 
  X_imp.combine = bnstruct::knn.impute(X.combine, k = 5)
  
  X_imp.test_missing = X_imp.combine[1: nrow(X.test_missing),,drop=F] 
  row_idx = which(rowSums(is.na(X.test)) >0)  
  #row_id = which(rownames(X.test) %in%  rownames(X_imp.test_missing)) 
  
  X_imp.test = X.test[, ]
  #X_imp.test[row_id, ] = X_imp.test_missing
  X_imp.test[row_idx, ] = X_imp.test_missing 
  
  result = list("train" = X_imp.train[, ], "test" = X_imp.test)
  return(result)
}

missForest_run <- function(X.train, y.train, X.test, y.test){
  #a) on training set 
  # parameter to set here 
  X_imp.train = missForest(X.train)$ximp
  
  #b) on testing set 
  X.test_missing = as.matrix(X.test[rowSums(is.na(X.test)) > 0 ,,drop=F ]) 
  X.combine = rbind(X.test_missing, X.train) 
  X_imp.combine <- missForest(X.combine)$ximp 
  
  X_imp.test_missing  = X_imp.combine[1:nrow(X.test_missing), ] 
  row_idx = which(rowSums(is.na(X.test)) >0) 
  #row_id = which(rownames(X.test) %in%  rownames(X_imp.test_missing)) 
  
  X_imp.test = X.test[, ]
  X_imp.test[row_idx, ] = X_imp.test_missing
  result = list("train" =  X_imp.train[, ],  "test" = X_imp.test[, ])
  return(result)
} 
