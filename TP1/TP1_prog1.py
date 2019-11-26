# import library
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.datasets import fetch_openml

def load_data(dataname):
    ds = fetch_openml(dataname)
    print(ds.DESCR)
    print("============================")
    print("Shape of data : ", ds.data.shape)
    print("Shape of target : ",ds.target.shape)
    return ds.data, ds.target


def main(): 
    data, target = load_data(DATA_NAME)
    target_list = np.unique(target)
    print("List of targets : ", target_list)

    images = data.reshape((-1,28,28))
    print("======= Plot random data ==========")

    plt.figure(figsize=(12,8))
    for i in range(10):
        idx = np.random.randint(len(data))
        plt.subplot(2,5,i+1)
        plt.imshow(images[idx],cmap=plt.cm.gray_r,interpolation="nearest")
        plt.title("Target : {}".format(target[idx]))
    plt.show()  
    plt.savefig("random-training-data.png")
    print("Image saved as random-training-data.png")
    

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=False,  default='mnist_784',\
                help="name of dataset from openML, type = string (available here : 'mnist_784', 'Fashion-MNIST')")
    args = vars(ap.parse_args())
    print("[DOC] -h for help")
    DATA_NAME = args['dataset']
    print("Explore dataset", DATA_NAME)
    print("===================================")
    main()
    print("=============End===================")
