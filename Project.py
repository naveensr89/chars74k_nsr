__author__ = 'naveen'
# Libs import
import numpy as np
import time

from sklearn import svm, preprocessing
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# File imports
from feature_expansion import feature_exp,feature_threshold
from grid_search import  grid_search
from utils import tile,read_X,read_y,save_out,print_accuracy

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
# tile(600,600,10,10,X_tr_orig,y_tr_orig, labels_string)
# tile(600,600,10,10,'train',y_tr_orig, labels_string,1)

###### Split train and test #########
X_tr, X_te, y_tr, y_te, sorted_files_tr, sorted_files_te = \
    train_test_split(X_tr_orig, y_tr_orig,sorted_files_tr_orig, test_size=0.33, random_state=42)

N_tr = X_tr.shape[0]
N_te = X_te.shape[0]
D = X_tr.shape[1]
print "Training : [Inputs x Features ] = [%d x %d]" % (N_tr,D)
print "Test     : [Inputs x Features ] = [%d x %d]" % (N_te,D)
print "\n "

####################### Feature Expansion ################################
X_tr = feature_exp(X_tr)
X_te = feature_exp(X_te)
D = X_tr.shape[1]
print "After Feature Expansion: Training : [Inputs x Features ] = [%d x %d]" % (N_tr,D)
print "After Feature Expansion: Test     : [Inputs x Features ] = [%d x %d]" % (N_te,D)
print "\n "

###################### Normalizing data ##################################
scaler = preprocessing.StandardScaler().fit(X_tr)
X_tr_n = scaler.transform(X_tr)
X_te_n = scaler.transform(X_te)

start = time.time()
print time.ctime()

########################### List of Classifiers #################################
# c_svm
# c_svm_l1
# log_reg
# c_svm_param
# knn
# naive_bayes
# ols
# ridge_reg
# lasso
# adaboost

classifier = "c_svm_param"

if(classifier == "c_svm"):
    ###################### C SVM - Accuracy - 0.44503 #############################
    model = svm.SVC()
    model.fit(X_tr_n, y_tr)
    y_tr_p = model.predict(X_tr_n)
    y_te_p = model.predict(X_te_n)
    # save_out(y_te_p,labels_string,sorted_files_te,'testLabels_CSVM.csv')

elif(classifier == "c_svm_l1"):
    ###################### C SVM L1 - Accuracy - 0.44503 #############################
    model = svm.LinearSVC(penalty='l1',dual=False)
    model.fit(X_tr_n, y_tr)
    y_tr_p = model.predict(X_tr_n)
    y_te_p = model.predict(X_te_n)

elif(classifier == "log_reg"):
    ###################### Logistic regression #############################
    model = linear_model.LogisticRegression()
    model.fit(X_tr_n, y_tr)
    y_tr_p = model.predict(X_tr_n)
    y_te_p = model.predict(X_te_n)

elif(classifier == "c_svm_param"):
    ###################### C SVM Param - Accuracy - 0.50164 #############################
    # grid_search(X_tr_n,y_tr)

    model = svm.SVC(C=10,kernel='rbf',gamma=0.001)
    model.fit(X_tr_n, y_tr)
    y_tr_p = model.predict(X_tr_n)
    y_te_p = model.predict(X_te_n)
    # save_out(y_te_p,labels_string,sorted_files_te,'testLabels_CSVM_Param_feature_exp.csv')

elif(classifier == "knn"):
    ###################### KNN - Accuracy -  #############################
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_tr_n, y_tr)
    y_tr_p = model.predict(X_tr_n)
    y_te_p = model.predict(X_te_n)
    # save_out(y_te_p,labels_string,sorted_files_te,'testLabels_kNN_feature_exp.csv')

elif(classifier == "naive_bayes"):
    ###################### Naive Bayes - Accuracy -  #############################
    model = GaussianNB()
    model.fit(X_tr_n, y_tr)
    y_tr_p = model.predict(X_tr_n)
    y_te_p = model.predict(X_te_n)

elif(classifier == "ols"):
    ###################### OLS - Accuracy -  #############################
    model = linear_model.LinearRegression()
    model.fit(X_tr_n,y_tr)
    y_tr_p = model.predict(X_tr_n)
    y_tr_p = np.round(y_tr_p)
    y_te_p = model.predict(X_te_n)
    y_te_p = np.round(y_te_p)

elif(classifier == "ridge_reg"):
    ###################### Ridge Regression - Accuracy -  #############################
    model = linear_model.Ridge(alpha=0.001)
    model.fit(X_tr_n,y_tr)
    y_tr_p = model.predict(X_tr_n)
    y_tr_p = np.round(y_tr_p)
    y_te_p = model.predict(X_te_n)
    y_te_p = np.round(y_te_p)

elif(classifier == "lasso"):
    ###################### Lasso - Accuracy -  #############################
    model = linear_model.Lasso(alpha=.15,max_iter=-1)
    model.fit(X_tr_n,y_tr)
    y_tr_p = model.predict(X_tr_n)
    y_tr_p = np.round(y_tr_p)
    y_te_p = model.predict(X_te_n)
    y_te_p = np.round(y_te_p)

elif(classifier == "adaboost"):
    ###################### AdaBoost ###########################################
    model = AdaBoostClassifier(KNeighborsClassifier(),n_estimators=100,algorithm='SAMME')
    model.fit(X_tr_n,y_tr)
    y_tr_p = model.predict(X_tr_n)
    y_te_p = model.predict(X_te_n)


print "\n"
print_accuracy(y_tr, y_tr_p, "Training")
print_accuracy(y_te, y_te_p, "Test")

end = time.time()
print "\nTime taken in sec %f" % (end-start)

