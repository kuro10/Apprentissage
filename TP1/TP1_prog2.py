# import library
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNNs
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import itertools
import time
import argparse


def load_data(dataname):
    ds = fetch_openml(dataname)
    print(ds.DESCR)
    print("============================")
    print("Shape of data : ", ds.data.shape)
    print("Shape of target : ",ds.target.shape)
    return ds.data, ds.target

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    """
    This function plots the confusion matrix of classfier
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

def main(): 
    data, target = load_data(DATA_NAME)
    target_list = np.unique(target)
    print("List of targets : ", target_list)
    images = data.reshape((-1,28,28))
    
    print("============================")
    print("Take a random batch of 5000 samples")
    indexes = random.sample(range(70000),5000)
    X,y  = data[indexes], target[indexes]
    imgs = images[indexes]  
    
    print("Splitting data.....")

    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size =0.8)
    print("Training data shape : ", X_train.shape)
    print("Training target shape : ", y_train.shape)
    print("Testing data shape : ", X_test.shape)
    print("Testing target shape : ", y_test.shape) 
       
    print("============================")
    print("Creating a k-NN classifier with k = 10")
    n_neighbors = 10
    clf = KNNs(n_neighbors)
    print(clf)
    print("Fitting the classifier...")
    start = time.clock()
    clf.fit(X_train,y_train)
    end = time.clock()
    print("CPU times : {} ms".format((end - start)*1000))
    
    print("============================")
    print("Plot some predictions...")
    print("Press Q to continue...")
    plt.figure(figsize=(12,8))
    for i in range(10):
        idx = np.random.randint(len(X_test))
        y_pred = clf.predict([X_test[idx]])
        plt.subplot(2,5,i+1)
        plt.imshow(X_test[idx].reshape(28,28))
        plt.title("Prediction : {}".format(y_pred[0]))
    plt.show()
    
    print("============================")
    print("Computing score.... Prediction score : ",clf.score(X_test,y_test))
    print("Classification report :")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test,y_pred))

     
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(12,8))
    plot_confusion_matrix(cm, classes=target_list, normalize=True,
                          title='Normalized confusion matrix')
    print("Press Q to continue...")
    plt.show()

    
if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=False,  default='mnist_784',\
                help="name of dataset from openML, type = string (available here : 'mnist_784', 'Fashion-MNIST')")
    args = vars(ap.parse_args())
    print("[DOC] Insert -h for help")
    DATA_NAME = args['dataset']
    print("Training K-NN with Dataset", DATA_NAME)
    print("===================================")
    main()
    print("=============End===================")
