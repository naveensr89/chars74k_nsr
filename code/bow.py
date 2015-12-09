import os
import numpy as np
import cv2
from sklearn.cluster import  KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import time
from scipy.spatial.distance import mahalanobis

class bow:

    def __init__(self, kmeans_K):
        self.kmeans_K = kmeans_K
        self.km_obj = KMeans(init='k-means++', n_clusters=self.kmeans_K, n_init=3)

    def sift_features(self,X):
        sift  = cv2.xfeatures2d.SIFT_create()
        des_all = np.array([]).reshape(0,128)
        des_size = np.array([]).reshape(0,1)

        N = X.__len__()
        count =0
        for i in range(N):
            img = X[i]
            kp, des = sift.detectAndCompute(img,None)
            if kp.__len__() == 0:
                des = np.zeros((1,128))
                count = count + 1
            des_all = np.vstack((des_all, des))
            des_size = np.vstack((des_size, des.shape[0]))

        des_size = np.cumsum(des_size)
        print "Total images with zero keypoints = %d"%count
        return des_all, des_size.astype('int')

    def get_hist(self, c_idx, f_size, kmeans_K):
        h = np.zeros((f_size.shape[0], kmeans_K))

        idx = range(0, f_size[0])
        h[0, :] = np.bincount(c_idx[idx],minlength=kmeans_K) / float(f_size[0])


        for i in range(1, f_size.shape[0]):
            idx = range(f_size[i - 1], f_size[i])
            h[i, :] = np.bincount(c_idx[idx],minlength=kmeans_K) / float(f_size[i] - f_size[i-1])

        return h

    def fit_predict(self, X):
        des_all, des_size = self.sift_features(X)

        c_idx_train = self.km_obj.fit_predict(des_all)

        #histogram
        h_train = self.get_hist(c_idx_train, des_size, self.kmeans_K)

        return h_train

    def predict(self, X):
        des_all, des_size = self.sift_features(X)
        c_idx_test = self.km_obj.predict(des_all)
        h_test = self.get_hist(c_idx_test, des_size, self.kmeans_K)

        return h_test