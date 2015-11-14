__author__ = 'naveen'
# Libs import
import numpy as np
import time

from sklearn import svm, preprocessing
from sklearn import linear_model
from sklearn.tree import  DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# File imports
from feature_expansion import feature_exp,feature_threshold,feature_sobel, feature_moments
from grid_search import  grid_search
from utils import tile,read_X,read_y,save_out,print_accuracy
from tf import tensorFlowNN
from keras_CNN import  keras_CNN


load_from_folder = 0
display_collage = 0
submission = 1

# classifier = "none"
classifier = "nn"

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
# voting
# random_forest

start = time.time()
print time.ctime()

if load_from_folder:
    ###### Read data and save to file ####
    X_tr_orig, sorted_files_tr_orig = read_X('../data/trainResized')
    X_te_orig, sorted_files_te_orig = read_X('../data/testResized')
    y_tr_orig,labels_string = read_y('../data/trainLabels.csv')

    np.save('../other/X_tr_orig',X_tr_orig)
    np.save('../other/X_te_orig',X_te_orig)
    np.save('../other/y_tr_orig',y_tr_orig)
    np.save('../other/labels_string',labels_string)
    np.save('../other/sorted_files_tr_orig',sorted_files_tr_orig)
    np.save('../other/sorted_files_te_orig',sorted_files_te_orig)

else:
    ###### Read data from saved file ####
    X_tr_orig = np.load('other/X_tr_orig.npy')
    X_te_orig = np.load('other/X_te_orig.npy')
    y_tr_orig = np.load('other/y_tr_orig.npy')
    labels_string = np.load('other/labels_string.npy')
    sorted_files_tr_orig = np.load('other/sorted_files_tr_orig.npy')
    sorted_files_te_orig = np.load('other/sorted_files_te_orig.npy')

if display_collage:
    ####################### Display Random images from Training set as collage  ######################
    tile(600,600,10,10,X_tr_orig,y_tr_orig, labels_string)
    tile(600,600,10,10,'data/train',y_tr_orig, labels_string,1)

if submission != 1:
    ###### Split train and test #########
    X_tr, X_te, y_tr, y_te, sorted_files_tr, sorted_files_te = \
        train_test_split(X_tr_orig, y_tr_orig,sorted_files_tr_orig, test_size=0.33, random_state=42)
else:
    X_tr, X_te, y_tr, sorted_files_tr, sorted_files_te = \
        X_tr_orig, X_te_orig, y_tr_orig, sorted_files_tr_orig, sorted_files_te_orig

N_tr = X_tr.shape[0]
N_te = X_te.shape[0]
D = X_tr.shape[1]
print "Training : [Inputs x Features ] = [%d x %d]" % (N_tr,D)
print "Test     : [Inputs x Features ] = [%d x %d]" % (N_te,D)

####################### Feature Expansion ################################
if classifier!="nn":
    X_tr = feature_exp(X_tr)
    X_te = feature_exp(X_te)
D = X_tr.shape[1]

print "After Feature Expansion: Training : [Inputs x Features ] = [%d x %d]" % (N_tr,D)
print "After Feature Expansion: Test     : [Inputs x Features ] = [%d x %d]" % (N_te,D)

###################### Normalizing data ##################################
scaler = preprocessing.StandardScaler().fit(X_tr)
X_tr_n = scaler.transform(X_tr)
X_te_n = scaler.transform(X_te)


end = time.time()
print "\nTime taken for Data preparation = %f sec" % (end-start)

start = time.time()
print time.ctime()

classifier_id = 0

if(classifier == "c_svm"):
    ###################### C SVM - Accuracy - 0.44503 #############################
    model = svm.SVC()
    model.fit(X_tr_n, y_tr)
    y_tr_p = model.predict(X_tr_n)
    y_te_p = model.predict(X_te_n)
    # save_out(y_te_p,labels_string,sorted_files_te,'submission/testLabels_CSVM.csv')

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
    best_params = grid_search(X_tr_n,y_tr)
    print best_params

    model = svm.SVC(C=best_params['C'],kernel=best_params['kernel'],gamma=best_params['gamma'])
    model.fit(X_tr_n, y_tr)
    y_tr_p = model.predict(X_tr_n)
    y_te_p = model.predict(X_te_n)

elif(classifier == "knn"):
    ###################### KNN - Accuracy -  #############################
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_tr_n, y_tr)
    y_tr_p = model.predict(X_tr_n)
    y_te_p = model.predict(X_te_n)

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
    model = AdaBoostClassifier(KNeighborsClassifier(n_neighbors=1),n_estimators=200,learning_rate=1, algorithm="SAMME")
    model.fit(X_tr_n,y_tr)
    y_tr_p = model.predict(X_tr_n)
    y_te_p = model.predict(X_te_n)

# elif(classifier == "voting"):
    # clf1 = DecisionTreeClassifier(max_depth=4)
    # clf2 = KNeighborsClassifier(n_neighbors=7)
    # clf3 = SVC(kernel='rbf', probability=True)
    # model = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2,1,2])
    # model.fit(X_tr_n,y_tr)
    # y_tr_p = model.predict(X_tr_n)
    # y_te_p = model.predict(X_te_n)

elif(classifier == "random_forest"):
    ###################### Random Forest ###########################################
    model =  RandomForestClassifier(n_estimators=1000,n_jobs=4)
    model.fit(X_tr_n,y_tr)
    y_tr_p = model.predict(X_tr_n)
    y_te_p = model.predict(X_te_n)

elif(classifier == "nn"):
    ############################### NN ###################################
    # tensorFlowNN(X_tr,y_tr,X_te,y_te)
    y_tr_p, y_te_p = keras_CNN(X_tr, y_tr, X_te)

else:
    print "No Classifier selected"
    classifier_id = -1



if classifier_id != -1:
    print "\n"
    print_accuracy(y_tr, y_tr_p, "Training")

    if submission != 1:
        print_accuracy(y_te, y_te_p, "Test")
    else:
        save_out(y_te_p,labels_string,sorted_files_te,'submission/testLabels_keras_CNN.csv')

end = time.time()
print "\nTime taken by classifier = %f sec" % (end-start)

