# import library
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

# import data
from sklearn.datasets import fetch_openml


# create target_list
target_list = np.unique(mnist.target)
print(target_list)

def visualize_data(dataset): 
    print("Shape of data : ", dataset.data.shape)
    print("Shape of target : ",dataset.target.shape)
    target_list = np.unique(dataset.target)
    print(target_list)
    plt.figure(figsize=(12,8))
    for i in range(10):
        idx = np.random.randint(len(dataset.data))
        plt.subplot(2,5,i+1)
        plt.imshow(dataset.data[idx].reshape(28,28))
        plt.title("Target : {}".format(dataset.target[idx]))
    plt.show()

def main(): 
   mnist = fetch_openml('mnist_784')
   visualize_data(mnist)



