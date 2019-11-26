# import library
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.datasets import fetch_openml


def main(): 
    ds = fetch_openml(DATA_NAME)
    print(ds)
    print("Shape of data : ", ds.data.shape)
    print("Shape of target : ",ds.target.shape)


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=False, default='mnist_784',\
                help="name of dataset from openML")
    args = vars(ap.parse_args())
    
    DATA_NAME = args['dataset']
    main()
