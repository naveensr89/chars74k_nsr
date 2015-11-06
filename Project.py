__author__ = 'naveen'
# Libs import
from sklearn import metrics, svm, cross_validation, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils.testing import all_estimators
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# File imports
from feature_expansion import feature_exp,feature_threshold
from grid_search import  grid_search
from utils import tile,read_X,read_y

###### Read data and save to file ####
# X_tr_orig, sorted_files_tr_orig = read_X('trainResized')
# X_te_orig, sorted_files_te_orig = read_X('testResized')
# y_tr_orig,labels_string = read_y('trainLabels.csv')

# np.save('X_tr_orig',X_tr_orig)
# np.save('X_te_orig',X_te_orig)
# np.save('y_tr_orig',y_tr_orig)
# np.save('labels_string',labels_string)
# np.save('sorted_files_tr_orig',sorted_files_tr_orig)
# np.save('sorted_files_te_orig',sorted_files_te_orig)

###### Read data from saved file ####
X_tr_orig = np.load('X_tr_orig.npy')
X_te_orig = np.load('X_te_orig.npy')
y_tr_orig = np.load('y_tr_orig.npy')
labels_string = np.load('labels_string.npy')
sorted_files_tr_orig = np.load('sorted_files_tr_orig.npy')
sorted_files_te_orig = np.load('sorted_files_te_orig.npy')

####################### Display Random images from Training set as collage  ######################
tile(600,600,10,10,X_tr_orig,y_tr_orig, labels_string)
tile(600,600,10,10,'train',y_tr_orig, labels_string,1)

###### Split train and test #########
X_tr, X_te, y_tr, y_te, sorted_files_tr, sorted_files_te = \
    train_test_split(X_tr_orig, y_tr_orig,sorted_files_tr_orig, test_size=0.33, random_state=42)

N_tr = X_tr.shape[0]
N_te = X_te.shape[0]
D = X_tr.shape[1]
print "Training : [Inputs x Features ] = [%d x %d]" % (N_tr,D)
print "Test     : [Inputs x Features ] = [%d x %d]" % (N_te,D)

####################### Feature Expansion ################################
X_tr = feature_exp(X_tr)
X_te = feature_exp(X_te)
D = X_tr.shape[1]
print "After Feature Expansion: Training : [Inputs x Features ] = [%d x %d]" % (N_tr,D)
print "After Feature Expansion: Test     : [Inputs x Features ] = [%d x %d]" % (N_te,D)


###################### Normalizing data ##################################
scaler = preprocessing.StandardScaler().fit(X_tr)
X_tr_n = scaler.transform(X_tr)
X_te_n = scaler.transform(X_te)

start = time.time()
print start
####################### C SVM - Accuracy - 0.44503 #############################
# clf = svm.SVC()
# clf.fit(X_tr_n, y_tr)
# y_tr_p = clf.predict(X_tr_n)
# print "Traning Classification Error = %f \%" % (sum(y_tr_p != y_tr)*100.0/N_tr)
# y_te_p = clf.predict(X_te_n)
# save_out(y_te_p,labels_string,sorted_files_te,'testLabels_CSVM.csv')

####################### C SVM L1 - Accuracy - 0.44503 #############################
# clf = svm.LinearSVC(penalty='l1',dual=False)
# clf.fit(X_tr_n, y_tr)
# y_tr_p = clf.predict(X_tr_n)
# print "Traning Classification Error = %f " % (sum(y_tr_p != y_tr)*100.0/N_tr)
# y_te_p = clf.predict(X_te_n)
# print "Test    Classification Error = %f " % (sum(y_te_p != y_te)*100.0/N_te)

####################### Logistic regression #############################
# clf = linear_model.LogisticRegression()
# clf.fit(X_tr_n, y_tr)
# y_tr_p = clf.predict(X_tr_n)
# print "Traning Classification Error = %f " % (sum(y_tr_p != y_tr)*100.0/N_tr)
# y_te_p = clf.predict(X_te_n)
# print "Test    Classification Error = %f " % (sum(y_te_p != y_te)*100.0/N_te)

####################### C SVM Param - Accuracy - 0.50164 #############################
# grid_search(X_tr_n,y_tr)

# clf = svm.SVC(C=10,kernel='rbf',gamma=0.001)
# clf.fit(X_tr_n, y_tr)
# y_tr_p = clf.predict(X_tr_n)
# y_te_p = clf.predict(X_te_n)
# #save_out(y_te_p,labels_string,sorted_files_te,'testLabels_CSVM_Param.csv')
# save_out(y_te_p,labels_string,sorted_files_te,'testLabels_CSVM_Param_feature_exp.csv')
# print "Traning Classification Error = %f " % (sum(y_tr_p != y_tr)*100.0/N_tr)
# print "Test    Classification Error = %f " % (sum(y_te_p != y_te)*100.0/N_te)

####################### KNN - Accuracy -  #############################
# neigh = KNeighborsClassifier(n_neighbors=5)
# neigh.fit(X_tr_n, y_tr)
# y_tr_p = neigh.predict(X_tr_n)
# y_te_p = neigh.predict(X_te_n)
# #save_out(y_te_p,labels_string,sorted_files_te,'testLabels_kNN_feature_exp.csv')
# print "Traning Classification Error = %f " % (sum(y_tr_p != y_tr)*100.0/N_tr)
# print "Test    Classification Error = %f " % (sum(y_te_p != y_te)*100.0/N_te)

####################### Naive Bayes - Accuracy -  #############################
# gnb = GaussianNB()
# gnb.fit(X_tr_n,y_tr)
# y_tr_p = gnb.predict(X_tr_n)
# y_te_p = gnb.predict(X_te_n)
# print "Traning Classification Error = %f " % (sum(y_tr_p != y_tr)*100.0/N_tr)
# print "Test    Classification Error = %f " % (sum(y_te_p != y_te)*100.0/N_te)

####################### OLS - Accuracy -  #############################
# clf = linear_model.LinearRegression()
# clf.fit(X_tr_n,y_tr)
# y_tr_p = clf.predict(X_tr_n)
# y_tr_p = np.round(y_tr_p)
# print "Traning Classification Error = %f " % (sum(y_tr_p != y_tr)*100.0/N_tr)

####################### Ridge Regression - Accuracy -  #############################
# clf = linear_model.Ridge(alpha=0.001)
# clf.fit(X_tr_n,y_tr)
# y_tr_p = clf.predict(X_tr_n)
# y_tr_p = np.round(y_tr_p)
# print "Traning Classification Error = %f " % (sum(y_tr_p != y_tr)*100.0/N_tr)

####################### Lasso - Accuracy -  #############################
# clf = linear_model.Lasso(alpha=.15,max_iter=-1)
# clf.fit(X_tr_n,y_tr)
# y_tr_p = clf.predict(X_tr_n)
# y_tr_p = np.round(y_tr_p)
# print "Traning Classification Error = %f " % (sum(y_tr_p != y_tr)*100.0/N_tr)

####################### Ridge Regression + polynomial features - Accuracy -  #############################
####################### Memory Error #####################################################################
# poly = preprocessing.PolynomialFeatures(interaction_only=True)
# X_tr_n_p = poly.fit_transform(X_tr_n,y_tr)
# clf = linear_model.Ridge(alpha=0.1)
# clf.fit(X_tr_n_p,y_tr)
# y_tr_p = clf.predict(X_tr_n_p)
# y_tr_p = np.round(y_tr_p)
# print "Traning Classification Error = %f " % (sum(y_tr_p != y_tr)*100.0/N_tr)

####################### AdaBoost #####################################################################
# clf = AdaBoostClassifier(KNeighborsClassifier(),n_estimators=100,algorithm='SAMME')
# clf.fit(X_tr_n,y_tr)
# y_tr_p = clf.predict(X_tr_n)
# y_te_p = clf.predict(X_te_n)
# print "Traning Classification Error = %f " % (sum(y_tr_p != y_tr)*100.0/N_tr)
# print "Test    Classification Error = %f " % (sum(y_te_p != y_te)*100.0/N_te)


end = time.time()
print "Time taken in sec %f" % (end-start)

