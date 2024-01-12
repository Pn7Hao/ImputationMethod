packages <- c(
  "caTools", 
  "glue", 
  "jsonlite", 
  "future.apply", 
  "progress", 
  "parallel"
)
# Install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}
# Packages loading
invisible(lapply(packages, library, character.only = TRUE)) 



project_path <- Sys.getenv("dimv")
project_path
library(here) 


source(here('src/rscript/dpers.R')) 

Sys.getenv("HOME") 

impDi <- function(S, Xtest, threshold=0.1, nlargest=2){
  # if (workers==-1){
  #   plan(multisession, workers = detectCores())  
  # }
  #    
  # if(print_progression==TRUE){
  #   print("====start imputation")
  # }
  Xtest[is.na(Xtest)] <- NaN   
  Xpred_original = Xtest
  
  #not to apply the algorithm on the ZERO VARIANCE columns, just fill the with 0 / or mean value (which is also 0)
  non_zero_var = (which(diag(S)!=0)) 
  
  Xtest = Xpred_original[, non_zero_var, drop=F] 
  Xpred = Xpred_original[, non_zero_var, drop=F] 
  S = S[non_zero_var, non_zero_var, drop=F]
  

  missingCols =  which(colSums(is.na(Xtest)) > 0) 
  # print(missingCols)
  if (length(missingCols)==0){
    print("There is no missing value feature or all column have zeros variance")
    return(Xpred_original)
  }

  DIMV1feature <- function(f){
    # i = which(missingCols==f)  
    # setTxtProgressBar(pb, i)  
  
    
    setF = which(abs(S[,f]) >= threshold)  
    #setF : Col with high corr with f
    #Fos need to be exist setF need to have at least one pair with different missing pattern f
    naRows = which(is.na(Xtest[,f]))
    #if there only 1 col have high correlation with f, and it has same missing pattern with f then find 1 other column have the highest correlation with f (and of course  corr lower than threshold)
    ColInSetFExistAtLeast1SameMissingValueAsf <- function(setF, naRow){ #list of True False = 
      return(
             sum(!is.na(Xtest[naRow, setF,drop=F]))#number of cell have missing value same as f  in setF
      )
    } 

    expandFForThisSample <- function(s){
      if (flag==T){
        tempS = S; diag(tempS) <- NA
        samePatternFeatures = which(
            colSums(is.na(Xtest[s,,drop=F])) == 1 # column that also have missing features at s
            )
        tempS[samePatternFeatures, samePatternFeatures] <- NA;
        mostSimilarCols <- order(
                tempS[,f,drop=F],
                decreasing = TRUE)[seq_len(nlargest)];
        setF_added = arrayInd(
                mostSimilarCols,
                dim(tempS[,f,drop=F]),
                useNames = TRUE)[, 1] # get indice of the items to add to set F 
      }
      return(setF_added)
    }


    
    Df_row_pool = which(is.na(Xtest[, f])) 
    
    # after having setF and f, we then find similar missing pattern in each row 
    while (length(Df_row_pool) > 0){
      s = Df_row_pool[1]
      flag = (
              sum(ColInSetFExistAtLeast1SameMissingValueAsf(setF, s))==0
            )
      if (flag == T){
        setF_expanded = expandFForThisSample(s) 
      }else{
        setF_expanded = c()
      }


      Fos = intersect(which(!is.na(Xtest[s,,drop=F])), c(setF, setF_expanded))
      Fms = intersect(which(is.na(Xtest[s,,drop=F])), c(setF, setF_expanded)) 
      
      #6 lines below is to calculate Z 
      maskOfXtestFilterFos = matrix(0, dim(Xtest)[1], dim(Xtest)[2]) 
      maskOfXtestFilterFos[Df_row_pool, Fos] = Xtest[Df_row_pool, Fos] 
      Z_row_ids_fiter_observed = intersect(which(rowSums(is.na(maskOfXtestFilterFos[, Fos,drop=F]))==0), Df_row_pool)
      maskOfXtestFilterFms = matrix(0, dim(Xtest)[1], dim(Xtest)[2])
      maskOfXtestFilterFms[Z_row_ids_fiter_observed, Fms] = Xtest[Z_row_ids_fiter_observed, Fms] 
      Z_row_ids_fiter_missing = which(rowSums(is.na(maskOfXtestFilterFms))==length(Fms))  
      
      Z_row_ids = Z_row_ids_fiter_missing
      
      So = S[Fos, Fos]
      Smo = S[Fos, f] 
      beta = solve(So) %*% Smo
      
      Xtest[Z_row_ids, f] = t(beta) %*% t(Xtest[Z_row_ids, Fos, drop=F])
      Df_row_pool <- setdiff(Df_row_pool, Z_row_ids)  
      
    }
    return(Xtest[, f])
  }
  
  total = length(missingCols)
  
  
  # if (print_progression==T){
  #   print(paste0("Number of missing column ", total))
  #   pb <- txtProgressBar(min = 0, max = total, style = 3)  
  # }
  
  
  # c = 0 
  # run the DIMV1feature parallely on every single features  
  # if (run_parallel == TRUE){
  Xpred_result= future_lapply(missingCols, DIMV1feature)
  Xpred[, missingCols] = t(do.call(rbind, Xpred_result)) 
  # }else{
  #   for (col in missingCols){
  #         result = DIMV1feature(col)
  #         if (c==0){
  #           results = result
  #         }else{
  #           results = rbind(results, result)
  #         }
  #         c=c+1
  #   }
  #   Xpred_result = results
  #   print(dim(Xpred_result))
  #   Xpred[, missingCols]  = Xpred_result
  # }
  # 
  
  
  Xpred_original[, non_zero_var] = Xpred
  Xpred_original[is.nan(Xpred_original)]=0
  
  return(Xpred_original) 
}   


