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
 
rmse_calc <- function(ori_data, imputed_rescaled_data,missing_pos_filter){
  nominator = sum((missing_pos_filter * ori_data - missing_pos_filter * imputed_rescaled_data)**2)
  denominator = sum(missing_pos_filter) 
  return(sqrt(nominator/denominator))
}
