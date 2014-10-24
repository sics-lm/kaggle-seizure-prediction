"""Module for doing the training of the models."""
from __future__ import division

import sklearn
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble

from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import numpy as np

import correlation_convertion
import dataset



def get_model(method, training_data_x, training_data_y):
    """
    Returns a dictionary with the model and cross-validation parameter grid for the model named *method*.
    """
    param_grid=dict()
    min_c = sklearn.svm.l1_min_c(training_data_x, training_data_y, loss='log')

    if method == 'logistic':
        clf = sklearn.linear_model.LogisticRegression(C=1, random_state=1729)
        param_grid = {'C': np.linspace(min_c, 1e5, 10), 'penalty': ['l1', 'l2'] }

    elif method == 'svm':
        clf = sklearn.svm.SVC(probability=True, class_weight='auto')
        param_grid =  [{'kernel': ['rbf'], 'gamma': [0, 1e-1, 1e-2, 1e-3],
                        'C': np.linspace(min_c, 1000, 3)}]

    elif method == 'sgd':
        clf = sklearn.linear_model.SGDClassifier()
        param_grid = [{'loss' : ['hinge', 'log'],
                       'penalty' : ['l1', 'l2', 'elasticnet'],
                       'alpha' : [0.0001, 0.001, 0.01, 0.1]}]

    elif method == 'random-forest':
        clf = sklearn.ensemble.RandomForestClassifier()

    else:
        raise NotImplementedError("Method {} is not supported".format(method))

    return dict(estimator=clf, param_grid=param_grid)


def get_cv_generator(training_data, do_segment_split=True):
    """
    Returns a cross-validation generator.
    """
    k_fold_kwargs = dict(n_folds=10, random_state=1729)
    if do_segment_split:
        cv = dataset.SegmentCrossValidator(training_data, cross_validation.StratifiedKFold, **k_fold_kwargs)
    else:
        cv = sklearn.cross_validation.StratifiedKFold(training_data['Preictal'], **k_fold_kwargs)


def train_model(interictal,
                preictal,
                method='logistic',
                training_ratio=0.8,
                do_downsample=True,
                do_segment_split=True,
                processes=1):
    training_data, test_data = dataset.split_experiment_data(interictal,
                                                             preictal,
                                                             training_ratio=training_ratio,
                                                             do_downsample=do_downsample,
                                                             do_segment_split=do_segment_split)
    test_data_x = test_data.drop('Preictal', axis=1)
    test_data_y = test_data['Preictal']

    clf = select_model(training_data, method=method,
                       training_ratio=training_ratio,
                       do_segment_split=do_segment_split,
                       processes=processes)

    print_report(clf, test_data_x, test_data_y)
    return clf


def fit_model(interictal, preictal, clf, do_downsample=True, downsample_ratio=2.0, do_segment_split=True):
    """
    Fits the classifier *clf* to the given preictal and interictal
    data. If *do_downsample* is true, the majority class will be
    downsampled before fitting. *downsample_ratio* gives the ratio of
    how the majority class to the minority class after downsampling.
    """

    training_data = dataset.merge_interictal_preictal(interictal, preictal,
                                                      do_downsample=do_downsample,
                                                      downsample_ratio=downsample_ratio,
                                                      do_segment_split=do_segment_split)


def print_report(clf, test_data_x, test_data_y):
    """
    Prints a report of how the classifier *clf* does on the test data.
    """

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    test_data_y_pred = clf.predict(test_data_x)
    print(classification_report(test_data_y, test_data_y_pred))
    print()
    print_cm(confusion_matrix(test_data_y, test_data_y_pred),
             labels=['Interictal', 'Preictal'])
    print()

def select_model(training_data, method='logistic',
                training_ratio=0.8,
                do_segment_split=True,
                processes=1):
    """Fits a model given by *method* to the training data."""
    print("Training a {} model".format(method))

    training_data_x = training_data.drop('Preictal', axis=1)
    training_data_y = training_data['Preictal']

    cv = get_cv_generator(training_data, do_segment_split=do_segment_split)

    model_dict = get_model(method, training_data_x, training_data_y)
    common_cv_kwargs = dict(cv=cv, scoring='roc_auc', n_jobs=processes, pre_dispatch='2*n_jobs', refit=True)

    cv_kwargs = dict(common_cv_kwargs)
    cv_kwargs.update(model_dict)

    clf = GridSearchCV(**cv_kwargs)
    clf.fit(training_data_x, training_data_y)

    return clf


def preictal_ratio(predictions):
    """Returns the ratio of 'Preictal' occurances in the dataframe *predictions*"""
    is_interictal = predictions == 'Preictal'  # A dataframe with Bools in the class column
    return is_interictal.sum() / is_interictal.count()


def assign_segment_scores(test_data, regr):
    """
    Returns a data frame with the segments of *test_data* as indices
    and the ratio of preictal guesses as a 'Preictal' column
    """

    predictions = regr.predict(test_data)
    df_predictions = pd.DataFrame(predictions,
                                  index=test_data.index,
                                  columns=('preictal',))
    segment_groups = df_predictions.groupby(level='segment')
    return segment_groups.mean()


def print_cm(cm, labels):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels])
    print("Colums show what the true values(rows) were classified as.")
    # Print header
    print(" " * columnwidth, end="\t")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end="\t")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("%{0}s".format(columnwidth) % label1, end="\t")
        for j in range(len(labels)):
            print("%{0}d".format(columnwidth) % cm[i, j], end="\t")
        print()
