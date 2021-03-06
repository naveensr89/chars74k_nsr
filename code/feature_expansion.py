__author__ = 'naveen'
import numpy as np
import cv2
from skimage.feature import hog

def feature_exp(X):
    # X_scales = feature_scales(X)

    #X_sums = feature_sums(X)
    # X_th = feature_threshold(X)
    # X_s = feature_sobel(X)
    # X_m = feature_moments(X)

    X_hog = feature_HOG(X)
    # X_hog_ms = feature_HOG_multiscale(X)
    # X_new = np.concatenate((X,X_sums,X_scales, X_th, X_s),axis=1)
    #X_new = np.concatenate((X_m, X_sums),axis=1) #X_th, X_s,
    X_new = X_hog
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
    T = 127
    X_add = np.uint8((X > T)*255);
    return X_add

def feature_sobel(X):
    N = X.shape[0]
    d = np.sqrt(X.shape[1])
    X_add = np.zeros((N,d*d))
    for i in range(N):
        img = X[i,]
        img = img.reshape((d,d))
        img = cv2.Sobel(img,ddepth=cv2.CV_8U, dx=1,dy=1)
        X_add[i,] = img.ravel()
    return X_add

def feature_moments(X):
    N = X.shape[0]
    d = np.sqrt(X.shape[1])
    X_add = np.zeros((N,7))
    for i in range(N):
        img = X[i,]
        img = img.reshape((d,d))
        img = cv2.moments(img,False)
        #print img['nu20'], img['nu11'], img['nu02'], img['nu30'], img['nu21'], img['nu12'], img['nu03']
        X_add[i,] = np.array([img['nu20'], img['nu11'], img['nu02'], img['nu30'], img['nu21'], img['nu12'], img['nu03']])
    return X_add

def feature_HOG(X):
    N = X.shape[0]
    d = np.sqrt(X.shape[1])
    # X_add = np.zeros((N,32))
    X_add = np.zeros((N,288))
    # Setup HOG descriptor
    for i in range(N):
        img = X[i,]
        img = img.reshape((d,d))
        # h = hog(img, orientations=8, pixels_per_cell=(10, 10),
        #             cells_per_block=(2, 2), visualise=False, normalise=True)
        h = hog(img, orientations=8, pixels_per_cell=(5, 5),
                    cells_per_block=(2, 2), visualise=False, normalise=True)
        X_add[i,] = h
    return X_add

def feature_HOG_multiscale(X):
    N = X.shape[0]
    d = np.sqrt(X.shape[1])
    # X_add = np.zeros((N,32))
    X_add = np.zeros((N,416))
    # Setup HOG descriptor
    for i in range(N):
        img = X[i,]
        img = img.reshape((d,d))
        h_1 = hog(img, orientations=8, pixels_per_cell=(5, 5),
                    cells_per_block=(2, 2), visualise=False, normalise=True)
        img = cv2.resize(img,(10,10))
        h_2 = hog(img, orientations=8, pixels_per_cell=(3, 3),
                    cells_per_block=(2, 2), visualise=False, normalise=True)
        h = np.hstack((h_1,h_2))
        h = h / np.sum(h)
        X_add[i,] = h
    return X_add
