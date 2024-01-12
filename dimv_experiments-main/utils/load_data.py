import gzip 
import os 
import numpy as np
import requests 
import argparse
from urllib.request import urlretrieve

def load_data(dataset_name):
    '''
    Args: 
    - dataset_name: ['fashion_mnist', 'mnist']
    Returns:
    - (Xtrain, ytrain, Xtest, ytest)
 
    '''    
    dataset_names = ["fashion_mnist", "mnist"]
    if dataset_name not in dataset_names:
        raise ValueError("Invalid dataset_name, Expected one of :%s " %dataset_names)
    urls = {
            "mnist": "http://yann.lecun.com/exdb/mnist/", 
            "fashion_mnist": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/" 
            }

    RESOURCES = [ 'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz']

    if (os.path.isdir('data') == 0):
        os.mkdir('data')
    if (os.path.isdir('data/{}'.format(dataset_name)) == 0):
        os.mkdir('data/{}'.format(dataset_name))
    data_path = 'data/{}/raw/'.format(dataset_name)
    if (os.path.isdir(data_path) == 0):
        os.mkdir(data_path)
    print(data_path) 
    for name in RESOURCES:
        print(data_path+name)
        if (os.path.isfile(data_path+name) == 0):
            url = urls.get(dataset_name)+name
            urlretrieve(url, os.path.join(data_path, name)) 
            print("Downloaded %s to %s" % (name, data_path))

    Xtrain, ytrain = load_mnist(data_path, kind = "train")
    Xtest, ytest = load_mnist(data_path, kind = 't10k')
    return Xtrain, ytrain, Xtest, ytest 



def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels 


