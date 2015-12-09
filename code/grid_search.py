__author__ = 'naveen'
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np

def grid_search(X_tr_n,y_tr):

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': np.logspace(-4, 2, 5),
                         'C': np.logspace(-3, 2, 6)}]
                        #{'kernel': ['linear'], 'C': np.logspace(-3, 2, 6)}]

    clf = GridSearchCV(SVC(), tuned_parameters, cv=3, n_jobs=4)
    clf.fit(X_tr_n, y_tr)

    return clf

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X_tr_n, y_tr, test_size=0.85, random_state=0)
    # scores = ['precision', 'recall']
    scores = ['recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=4,
                           scoring='%s_weighted' % score, n_jobs=4)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    return clf
