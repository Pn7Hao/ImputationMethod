#!/usr/bin/Rscript
require("datasets");
require("mvnTest");


# Read mnist dataset from .gz file
get_mnist = function(){
	# read training set only
	f = file("datasets/mnists/train-image", "rb");
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    X = readBin(f,'integer',n=n*nrow*ncol,size=1,signed=F)
    close(f);

	X = matrix(X, ncol=nrow*ncol, byrow=T);
	
	#Subseting 
	X = X[1:1000,];

	# count unique
	n_unique = apply(X, 2, function(x) {length(unique(x))});
	X = X[, n_unique > 1]

	return(X);
}


get_fashion_mnist = function(){
	# read training set only
	f = file("datasets/fashion_mnist/train-img", "rb");
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    X = readBin(f,'integer',n=n*nrow*ncol,size=1,signed=F)
    close(f);

	X = matrix(X, ncol=nrow*ncol, byrow=T);
	
	#Subseting 
	X = X[1:1000,];

	# count unique
	n_unique = apply(X, 2, function(x) {length(unique(x))});
	X = X[, n_unique > 1]
}

get_ds = function(ds) {
	if (ds == 'yeast') {
		url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data';
		df = read.csv(url,
					  header=F,
					  sep="")

		df = subset(df, select = -c(1)) 
		
		# X, y
		X = df[1:ncol(df) - 1]
		y = df[ncol(df)]

		return(list("X" = X, "y" = y));
	} 

	if (ds == "new_thyroid"){
		url = "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data"
		df = read.csv(url, header=F)	
		
		# X, y
		X = df[1:ncol(df) - 1]
		y = df[ncol(df)]

		return(list("X" = X, "y" = y));
	}

	if (ds == "iris") {
		data(iris); 

		# X, y
		X = iris[1:ncol(iris) - 1]
		y = iris[ncol(iris)]
		
		return(list("X" = X, "y" = y));
	}

	if (ds == "mnist") {
		X = get_mnist();

		return(list("X" = X, "y" = NULL));
	}
	
	if (ds == "fashion_mnist") {
		X = get_fashion_mnist();

		return(list("X" = X, "y" = NULL));
	}
};


main =  function(ds_name) {
	ds = get_ds(ds_name);
	print("Dataset:");
	print(ds_name);
	print("-------------------------");
	print(R.test(ds$X));
}

args = commandArgs(trailingOnly=TRUE);

if (length(args) != 1) {
	stop("Expected 1 argument: ds_name");
}

main(args[1]);
