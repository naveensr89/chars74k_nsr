__author__ = 'naveen'
import numpy as np
import cv2
from sklearn import metrics, svm, cross_validation, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import linear_model

def feature_exp(X):
    X_sums = feature_sums(X)
    X_scales = feature_scales(X)
    X_th = feature_threshold(X)
    X_new = np.concatenate((X,X_sums,X_scales, X_th),axis=1)
    return X_new

def feature_sums(X):
    N = X.shape[0]
    d = np.sqrt(X.shape[1])
    X_add = np.zeros((N,d*2+1))
    for i in range(N):
        img = X[i,]
        img = img.reshape((d,d))
        r_sum = img.sum(axis=1)
        c_sum = img.sum(axis=0)
        a_sum = c_sum + r_sum
        a_sum = np.sum(a_sum)
        X_add[i,] = np.hstack((r_sum,c_sum,a_sum))
    return X_add

def feature_scales(X):
    N = X.shape[0]
    d = np.sqrt(X.shape[1])
    X_add = np.zeros((N,100+25+4))
    for i in range(N):
        img = X[i,]
        img = img.reshape((d,d))
        img2 = cv2.resize(img,(10,10)).flatten()
        img4 = cv2.resize(img,(5,5)).flatten()
        img8 = cv2.resize(img,(2,2)).flatten()
        X_add[i,] = np.hstack((img2,img4,img8))
    return X_add

def feature_threshold(X):
    N = X.shape[0]
    d = np.sqrt(X.shape[1])
    T = 127
    # X_add = np.zeros((N,d*d))
    # for i in range(N):
    #     img = (X[i,] > T)
    #     X_add[i,] = img
    X_add = np.uint8((X > T)*255);
    return X_add