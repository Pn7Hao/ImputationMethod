 packages <- c(
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

dpers <- function(Xscaled){
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

  find_cov_upper_triag = function(i) {

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
