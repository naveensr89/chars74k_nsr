
__author__ = 'naveen'
from sklearn import linear_model
from sklearn.tree import  DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

import numpy as np
from utils import print_accuracy,read_X_full_res
from grid_search import  grid_search
from tf import tensorFlowNN
from keras_CNN import  keras_CNN
from bow import bow

def models(X_tr_n, y_tr, X_te_n, classifier):
    if(classifier == "c_svm"):
        ###################### C SVM - Accuracy - 0.44503 #############################
        model = SVC()
        model.fit(X_tr_n, y_tr)
        y_tr_p = model.predict(X_tr_n)
        y_te_p = model.predict(X_te_n)
        # save_out(y_te_p,labels_string,sorted_files_te,'submission/testLabels_CSVM.csv')

    elif(classifier == "c_svm_l1"):
        ###################### C SVM L1 - Accuracy - 0.44503 #############################
        model = LinearSVC(penalty='l1',dual=False)
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
        model = grid_search(X_tr_n,y_tr)
        print "Best params = "
        print model.best_params_

        # model = SVC(C=10,kernel='rbf',gamma=0.001)
        # model.fit(X_tr_n, y_tr)
        y_tr_p = model.predict(X_tr_n)
        y_te_p = model.predict(X_te_n)

    elif(classifier == "knn"):
        ###################### KNN - Accuracy -  #############################
        model = KNeighborsClassifier(n_neighbors=20)
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
        # model = AdaBoostClassifier(RandomForestClassifier(max_features=50, n_estimators=10, max_depth=20),
        #                            n_estimators=100,learning_rate=2)
        model = AdaBoostClassifier(linear_model.SGDClassifier(n_iter=50),n_estimators=100,learning_rate=1, algorithm="SAMME")
        # model = AdaBoostClassifier(n_estimators=100,learning_rate=2)
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
        # model =  RandomForestClassifier(n_estimators=100,n_jobs=4)

        # Grid search
        clf =  RandomForestClassifier(n_jobs=3)
        param_grid = {"max_depth": [10, 20, 30],
                      "max_features": [50, 100, 200],
                      "n_estimators": [10,50,100]}

        # run grid search
        model = GridSearchCV(clf, param_grid=param_grid)
        model.fit(X_tr_n,y_tr)

        print model.best_params_

        y_tr_p = model.predict(X_tr_n)
        y_te_p = model.predict(X_te_n)

    elif(classifier == "nn"):
        ############################### NN ###################################
        # tensorFlowNN(X_tr,y_tr,X_te,y_te)
        y_tr_p, y_te_p = keras_CNN(X_tr, y_tr, X_te)

    elif(classifier == "bow"):
        ############################### BOW ###################################
        X_tr_full_res, s = read_X_full_res('data/train')
        X_te_full_res, s = read_X_full_res('data/test')

        bow_obj = bow(kmeans_K = 100)
        X_bow_tr = bow_obj.fit_predict(X_tr_full_res)
        X_bow_te = bow_obj.predict(X_te_full_res)

        model = SVC()
        model.fit(X_bow_tr, y_tr)
        y_tr_p = model.predict(X_bow_tr)
        y_te_p = model.predict(X_bow_te)

    else:
        print "No Classifier selected"
        return False


    print_accuracy(y_tr, y_tr_p, "Training")

    return y_te_p
