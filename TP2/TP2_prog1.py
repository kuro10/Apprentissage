# import library
import numpy as np
import matplotlib.pyplot as plt
import time, random 
import argparse
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve, zero_one_loss

def load_data(dataname):
    ds = fetch_openml(dataname)
    print(ds.DESCR)
    data, target = ds.data, ds.target
    print("============================")
    print("Shape of data : ", ds.data.shape)
    print("Shape of target : ",ds.target.shape)
    target_list = np.unique(ds.target)
    print("List of targets : ", target_list)
    ## Normalize data  
    data = data / 255.
    X_train,X_test,y_train,y_test = train_test_split(data,target,train_size=0.7)
    print("Traning data shape : ", X_train.shape)
    print("Traning target shape : ", y_train.shape)
    print("Testing data shape : ", X_test.shape)
    print("Testing target shape : ", y_test.shape)
    
    return X_train,X_test,y_train,y_test


def main(): 
    ## load & normalize & split data
    X_train,X_test,y_train,y_test = load_data(dataname=DATA_NAME)
    
    ## build a simple MLP
    mlp = MLPClassifier(hidden_layer_sizes=50,max_iter=30,verbose=1)
    
    ## training model
    print("============================")
    print("Training model...")
    mlp.fit(X_train,y_train)
    print("Finish training.")
    print("Training accuracy: %f" % mlp.score(X_train, y_train))
    print("Training loss : %f" % zero_one_loss(y_train, mlp.predict(X_train)))
    print("Test accuracy: %f" % mlp.score(X_test, y_test))
    print("Test loss : %f" % zero_one_loss(y_test, mlp.predict(X_test)))

    
    ## predict 
    print("============================")
    print("Plot some predictions...")
    print("Press Q to continue...")
    plt.figure(figsize=(12,8))
    for i in range(10):
        idx = np.random.randint(len(X_test))
        y_pred = mlp.predict([X_test[idx]])
        plt.subplot(2,5,i+1)
        plt.imshow(X_test[idx].reshape(28,28))
        plt.title("Prediction : {}".format(y_pred[0]))
    plt.show()

    print("============================")
    print("Classification report :")
    y_pred = mlp.predict(X_test)
    print(classification_report(y_test,y_pred))
    
    

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
