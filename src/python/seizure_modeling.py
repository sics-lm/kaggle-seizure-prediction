"""Module for doing the training of the models."""
from __future__ import division

import sklearn
import sklearn.linear_model
import sklearn.svm
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

import pandas as pd
import numpy as np

import correlation_convertion
import dataset



def train_model(training_data, method='logistic',
                training_ratio=0.8,
                do_segment_split=True,
                processes=1):
    """Fits a model given by *method* to the training data."""
    print("Training a {} model".format(method))

    k_fold_kwargs = dict(n_folds=10, random_state=1729)
    if do_segment_split:
        cv = dataset.SegmentCrossValidator(training_data, cross_validation.StratifiedKFold, **k_fold_kwargs)
    else:
        cv = sklearn.cross_validation.StratifiedKFold(training_data['Preictal'], **k_fold_kwargs)

    common_kwargs = dict( cv=cv, scoring='roc_auc', n_jobs=processes, pre_dispatch='2*n_jobs', refit=True)
    if method == 'logistic':
        regr = sklearn.linear_model.LogisticRegression(C=1e5)
        param_grid = {'C': np.linspace(0, 1e5, 20) }
        scores = ['roc_auc']
        clf = GridSearchCV(estimator=regr, param_grid=param_grid, **common_kwargs)

    elif method == 'svm':
        regr = sklearn.svm.SVC(C=1)
        param_grid =  [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                       {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        scores = ['roc_auc']
        clf = GridSearchCV(estimator=regr, param_grid=param_grid, **common_kwargs)

    #if method == 'linear':
    else:
        clf = sklearn.linear_model.LinearRegression()


    training_data_x = training_data.drop('Preictal', axis=1)
    training_data_y = training_data['Preictal']

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
