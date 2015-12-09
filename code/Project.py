__author__ = 'naveen'
import sys
sys.path.append('/home/naveen/Dropbox/MS/EE660/chars74k_nsr/code')

# Libs import
import numpy as np
import time

from sklearn import preprocessing

# File imports
from feature_expansion import feature_exp
from utils import save_out,print_accuracy, load
from models import models

load_from_folder = 0
display_collage = 0
submission = 0
submission_fname = 'submission/testLabels_HOG_multiscale_c_svm_param.csv'

classifier = "c_svm_param"

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
# bow

start = time.time()
print time.ctime()

X_tr, X_te, y_tr, y_te, sorted_files_tr, sorted_files_te = \
    load(load_from_folder, display_collage, submission)

N_tr = X_tr.shape[0]
N_te = X_te.shape[0]
D = X_tr.shape[1]
print "Training : [Inputs x Features ] = [%d x %d]" % (N_tr,D)
print "Test     : [Inputs x Features ] = [%d x %d]" % (N_te,D)

####################### Feature Expansion ################################
if classifier!="nn" and classifier!="bow":
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

y_te_p = models(X_tr_n, y_tr, X_te_n, classifier)

if isinstance(y_te_p,np.ndarray):
    if submission != 1:
        print_accuracy(y_te, y_te_p, "Test")
    else:
        save_out(y_te_p,labels_string,sorted_files_te,submission_fname)

end = time.time()
print "\nTime taken by classifier = %f sec" % (end-start)
