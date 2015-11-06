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

def tile(wd,ht,rows,cols,input,y_tr, labels_string, color=0):
    if(isinstance(input,str)):
        folder = input
        files = [f for f in os.listdir(folder) if f.endswith(".Bmp")]
        files = sorted(files, key=lambda x: int(x.split('.')[0]))
        numfiles = files.__len__()
        folder_passed = 1
    elif(type(input).__module__ == np.__name__):
        X = input
        numfiles = X.shape[0]
        folder_passed = 0
    else:
        print "Pass either folder or Input matrix X in input"
        return

    if folder_passed == 1 and color == 1:
        out = np.zeros((ht,wd,3),dtype=np.uint8)
        clr_idx = cv2.IMREAD_COLOR
    else:
        out = np.zeros((ht,wd),dtype=np.uint8)
        clr_idx = cv2.IMREAD_GRAYSCALE

    iwd = wd / cols
    iht = ht / rows

    # overlay Text font
    font = cv2.FONT_HERSHEY_TRIPLEX

    for row in range(rows):
        for col in range(cols):
            idx = random.randint(0,numfiles)
            label = labels_string[y_tr[idx]]
            if folder_passed == 1:
                fname = os.path.splitext(files[idx])[0]
                # print idx,files[idx]
                # print fname,label
                img = cv2.imread(folder+'/'+files[idx],clr_idx)
                img = cv2.resize(img,(iht,iwd),interpolation = cv2.INTER_CUBIC)
            else:
                img = X[idx,]
                d = np.sqrt(X.shape[1])
                img = img.reshape((d,d)) # Assuming square image
                img = cv2.resize(img,(iht,iwd),interpolation = cv2.INTER_LINEAR)

            cv2.putText(img,label,(0,3*iht/4), font, 1,(255,255,255),1,cv2.LINE_AA)

            if folder_passed == 1 and color == 1:
                out[row*iht:(row+1)*iht,col*iwd:(col+1)*iwd,] = img
            else:
                out[row*iht:(row+1)*iht,col*iwd:(col+1)*iwd] = img

    cv2.imshow('Collage',out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_X(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".Bmp")]
    sorted_files = sorted(files, key=lambda x: int(x.split('.')[0]))
    f = sorted_files[0]
    img = cv2.imread(folder+'/'+f,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    img = img[:,:,1]
    wd = img.shape[0]
    ht = img.shape[1]
    N = files.__len__()
    print wd,ht,N
    X = np.zeros((N,wd*ht))
    for i in range(N):
        f = sorted_files[i]
        img = cv2.imread(folder+'/'+f,0)
        X[i,] = img.flatten()

    return X, sorted_files

def read_y(csv_file):
    # Reading Labels
    y_tr_df = pd.read_csv(csv_file,index_col = 'ID')
    y_tr_string = y_tr_df['Class'].as_matrix()

    # Converting Labels from String to Int
    labels_string = np.unique(y_tr_string)
    labels_int = range(labels_string.shape[0])
    labels_dict = dict(zip(labels_string,labels_int))

    y_tr = [labels_dict[y] for y in y_tr_string]
    return y_tr,labels_string

# Format is in Kaggle submission format
# Save test results to csv file
def save_out(y_te,labels_string, sorted_files, file_name):
    y_te_string = [labels_string[y] for y in y_te]
    ids = [f.split('.')[0] for f in sorted_files]
    y_df = pd.DataFrame({'Class': y_te_string, 'ID':ids})
    y_df.set_index('ID')
    y_df.to_csv(file_name,index=False, columns=['ID','Class'])
    return

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
# tile(600,600,5,5,'train',y_tr_orig, labels_string,1)

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

