# FLOW TO RUN THE EXPERIMENT IN THE FOLDER
step 1: run splitting_folds.Rmd (by chunk in the notebook)
  - convert to .R file (if you prefer .R script) by running the chunk
    ```purl("splitting_folds.Rmd", output = 'splitting_folds.R')```
    and then run
    ```
    Rscript splitting_folds.R
    ```
  - change dataset in splitting_folds.Rmd file (if u want to run other datasets)
step 2: imputation
  - if you want to impute small dataset, run this code in terminal: 
  ```
  Rscript imputation.R 
  ```
  - if you want to impute mnist dataset (mnist/fashion mnist), run this code in terminal: 
  ```
  Rscript imputation_mnist.R 
  ```
  
  note that all .R file is converted from .Rmd directly, if you want to use both Rmd and R, change Rmd file and convert to .R file (by running the chunk ``` purl("Rmd_file_name.Rmd", output = "R_file_name.R")``` 


step 3: classification 

  run script in "classification.R " (SVM)
