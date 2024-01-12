all: download_mnist download_fashion_mnist unzip norm_test

norm_test:
	./norm_test.R iris >> norm_test.log
	./norm_test.R yeast >> norm_test.log
	./norm_test.R new_thyroid >> norm_test.log
	./norm_test.R mnist >> norm_test.log
	./norm_test.R fashion_mnist >> norm_test.log 


download_mnist:
	mkdir -p datasets
	mkdir -p datasets/mnists
	curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz	-o datasets/mnists/train-image.gz  

download_fashion_mnist:
	mkdir -p datasets
	mkdir -p datasets/fashion_mnist;
	echo [+] Downloading fashion mnist
	curl http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz -o datasets/fashion_mnist/train-img.gz

unzip:
	cd datasets/mnists; gzip -dk train-image.gz
	cd datasets/fashion_mnist; gzip -dk train-img.gz
