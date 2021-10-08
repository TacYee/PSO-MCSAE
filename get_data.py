import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plot
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
import scipy.io as scio


def get_X_train():
    data=scio.loadmat('metadata.mat')
    isgrasp=scio.loadmat('isGrasp.mat')
    X_train=[]
    for i in range(len(data['splitId'])):
        if data['splitId'][i]==0 and isgrasp['isGrasp'][i]==1:
            X_train.append(data['pressure'][i])
    X_train=np.array(X_train)
    scaler=StandardScaler().fit(X_train[0])
    X_train_rescaled=[]
    for i in range(len(X_train)):
        X_train_rescaled_i=scaler.transform(X_train[i])
        X_train_rescaled.append(X_train_rescaled_i)
    X_train_rescaled=np.array(X_train_rescaled)
    X_train_rescaled_=X_train_rescaled.reshape(5408,32,32,1)
    
    return X_train_rescaled_

def get_y_train():
    data=scio.loadmat('metadata.mat')
    isgrasp=scio.loadmat('isGrasp.mat')    
    y_train=[]
    for i in range(len(data['splitId'])):
        if  data['splitId'][i]==0 and isgrasp['isGrasp'][i]==1:
            y_train.append(data['objectId'][i])
    y_train_cate=to_categorical(y_train,num_classes=8)
    return y_train_cate

def get_X_test():
    data=scio.loadmat('metadata.mat')
    isgrasp=scio.loadmat('isGrasp.mat')
    X_test=[]
    for i in range(len(data['splitId'])):
        if data['splitId'][i]==1 and isgrasp['isGrasp'][i]==1:
            X_test.append(data['pressure'][i])
    X_test=np.array(X_test)
    scaler=StandardScaler().fit(X_test[0])
    X_test_rescaled=[]
    for i in range(len(X_test)):
        X_test_rescaled_i=scaler.transform(X_test[i])
        X_test_rescaled.append(X_test_rescaled_i)
    X_test_rescaled=np.array(X_test_rescaled)
    X_test_rescaled_=X_test_rescaled.reshape((2289,32,32,1))
    return X_test_rescaled_

def get_y_test():
    data=scio.loadmat('metadata.mat')
    isgrasp=scio.loadmat('isGrasp.mat')    
    y_test=[]
    for i in range(len(data['splitId'])):
        if  data['splitId'][i]==1 and isgrasp['isGrasp'][i]==1:
            y_test.append(data['objectId'][i])
    y_test_cate=to_categorical(y_test,num_classes=8)
    return y_test_cate

if __name__=='__main__':
    X_train=get_X_train()
    print(X_train.shape)
    X_test=get_X_test()
    print(X_test.shape)
