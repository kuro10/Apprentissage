# import library
import numpy as np
import matplotlib.pyplot as plt
import time 
import argparse

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import precision_score, classification_report, confusion_matrix
from sklearn.metrics import zero_one_loss


def load_data(dataname):
    ds = fetch_openml(dataname)
    print(ds.DESCR)
    data, target = ds.data, ds.target
    print("============================")
    print("Shape of data : ", ds.data.shape)
    print("Shape of target : ",ds.target.shape)
    target_list = np.unique(ds.target)
    print("List of targets : ", target_list)

    X_train,X_test,y_train,y_test = train_test_split(data,target,train_size=0.7)
    print("Training data shape : ", X_train.shape)
    print("Training target shape : ", y_train.shape)
    print("Testing data shape : ", X_test.shape)
    print("Testing target shape : ", y_test.shape)
    
    return X_train,X_test,y_train,y_test


def main(): 
    ## load & normalize & split data
    X_train,X_test,y_train,y_test = load_data(dataname=DATA_NAME)
    
    ## build a simple linear SVM
    clf = LinearSVC()
    
    ## training model
    print("============================")
    print("Training model...")
    start = time.clock()
    clf.fit(X_train,y_train)
    exec_time = time.clock() - start
    print("Finish training.")
    print("Execution time : {} ms".format(exec_time*1000))
    print("Training accuracy: %f" % clf.score(X_train, y_train))
    print("Training loss : %f" % zero_one_loss(y_train, clf.predict(X_train)))
    print("Test accuracy: %f" % clf.score(X_test, y_test))
    print("Test loss : %f" % zero_one_loss(y_test, clf.predict(X_test)))

    
    ## predict 
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
    print("Classification report :")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test,y_pred))
    
    print("Confusion matrix:\n%s" % confusion_matrix(y_test, y_pred))

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=False,  default='mnist_784',\
                help="name of dataset from openML, type = string (available here : 'mnist_784', 'Fashion-MNIST')")
    args = vars(ap.parse_args())
    print("[DOC] Insert -h for help")
    DATA_NAME = args['dataset']
    print("Explore dataset", DATA_NAME)
    print("===================================")
    main()
    print("=============End===================")
