 ackages <- c(
  "caTools", 
  "glue", 
  "jsonlite", 
  "future.apply", 
  "progress"
)
# Install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

# Packages loading
invisible(lapply(packages, library, character.only = TRUE)) 

plan(multisession, workers = 8)   



find_cov_ij <- function(Xij, Sii, Sjj){
  # Xij: the i, j column of the original matrix
  # sii, sjj = \hat{Sigma}_{ii}, \hat{Sigma}_{jj}
  # s11 = sum(Xij[,1]**2, na.rm = TRUE)
  # s12 = sum(Xij[,1]*Xij[,2], na.rm = TRUE)
  # s22 = sum(Xij[,2]**2, na.rm = TRUE)
  #start edited--------------------------
  comlt_Xij = Xij[complete.cases(Xij),,drop=F ] 
  s11 = sum(comlt_Xij[,1]**2)
  s12 = sum(comlt_Xij[,1]*comlt_Xij[,2])
  s22 = sum(comlt_Xij[,2]**2)
  #end edited-------------------------- 
  m = sum(complete.cases(Xij))
  coef = c(s12*Sii*Sjj, 
                 m*Sii*Sjj-s22*Sii-s11*Sjj,
                 s12, -m)
  sol = polyroot(z = coef)
  sol = Re(sol)
  scond = Sjj - sol^2/Sii
  
  #Sii >0 
  #start edited--------------------------
  #etas = suppressWarnings(-m*log(sol) - (Sjj-2*sol/Sii*s12+sol^2/Sii^2*s11)/scond)
  etas = suppressWarnings(-m*log(scond) - (Sjj-2*sol/Sii*s12+sol^2/Sii^2*s11)/scond) 
  #end edited--------------------------
  return(sol[which.max(etas)])
}

dpers <- function(Xscaled, show_progress=F){
  # Xscaled: scaled input with missing data
  # THE INPUT MUST BE NORMALIZED ALREADY
  shape = dim(Xscaled) # dimension
  S = matrix(0, shape[2],shape[2])
  diag(S) = apply(Xscaled, 2, function(x) var(x, na.rm=TRUE))
  # Get the index of the upper triangular matrix (row, column)
  Index<-which(upper.tri(S,diag=FALSE),arr.ind=TRUE)
  # compute the covariance and assign to S based on Index
  #start edited-------------------------- 
  total = nrow(Index)  
  if (show_progress==T){
    pb <- txtProgresSBar(min = 0, max = total, style = 3)   
  }
  find_cov_upper_triag = function(i) {
    if (show_progress==T){
      setTxtProgressBar(pb, i)
    }
    if (S[Index[i,1], Index[i,1]] == 0 | S[Index[i,2], Index[i,2]] == 0){
      return(NA)
    }
    else{
      return (
        find_cov_ij(
            Xscaled[,c(Index[i,1],Index[i,2])], 
            S[Index[i,1], Index[i,1]], 
            S[Index[i,2], Index[i,2]]
            )
      )
    }
  }
  numCores = availableCores() 
  plan(multisession, workers = numCores)  
  S_upper_calc = unlist(future_lapply(1:total, find_cov_upper_triag))
  
  stopifnot(length(S_upper_calc) == length(S[Index]))
  #end edited-------------------------- 
  S[Index]  = S_upper_calc 
  S = S + t(S)
  diag(S) = diag(S)/2
  return(S)
}  

impDi <- function(S, Xtest, threshold, nlargest=2){
  Xtest[is.na(Xtest)] <- NaN   
  Xpred_original = Xtest
  
  #not to apply the algorithm on the ZERO VARIANCE columns, just fill the with 0 / or mean value (which is also 0)
  non_zero_var = (which(diag(S)!=0)) 
  
  Xtest = Xpred_original[, non_zero_var, drop=F] 
  Xpred = Xpred_original[, non_zero_var, drop=F] 
  S = S[non_zero_var, non_zero_var, drop=F]
  

  missingCols =  which(colSums(is.na(Xtest)) > 0) 

  if (length(missingCols)==0){
    print("There is no missing value feature")
    return(Xpred_original)
  }

  DIMV1feature <- function(f){
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
    for (naRow in naRows){
      flag = (
              sum(ColInSetFExistAtLeast1SameMissingValueAsf(setF, naRow))==0
              )
      if (flag==T){
        tempS = S; diag(tempS) <- NA
        samePatternFeatures = which(
            colSums(is.na(Xtest[naRow,,drop=F])) == 1
        )
        tempS[samePatternFeatures, samePatternFeatures] <- NA; 
        mostSimilarCols <- order(
                tempS[,f,drop=F],
                decreasing = TRUE)[seq_len(nlargest)]; 
        setF_added = arrayInd(
                mostSimilarCols, 
                dim(tempS[,f,drop=F]), 
                useNames = TRUE)[, 1]
        setF = c(setF, setF_added) 
      }
    }

    Df_row_pool = which(is.na(Xtest[, f])) 
    
    # after having setF and f, we then find similar missing pattern in each row 
    while (length(Df_row_pool) > 0){
      s = Df_row_pool[1]
      Fos = intersect(which(!is.na(Xtest[s,,drop=F])), setF)
      Fms = intersect(which(is.na(Xtest[s,,drop=F])), setF) 
      
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
  # run the DIMV1feature parallely on every single features  
  Xpred_result= future_lapply(missingCols, DIMV1feature)
  Xpred[, missingCols] = t(do.call(rbind, Xpred_result)) 
  
  Xpred_original[, non_zero_var] = Xpred
  Xpred_original[is.nan(Xpred_original)]=0
  
  return(Xpred_original) 
}   
 
