"""Module for doing the training of the models."""
from __future__ import division

import sklearn
import sklearn.linear_model
import sklearn.cross_validation
import pandas as pd

import correlation_convertion


def train_model(training_data, method='logistic',
                training_ratio=0.8,
                do_segment_split=True,
                processes=1):
    """Fits a model given by *method* to the training data."""

    if method == 'logistic':
        regr = sklearn.linear_model.LogisticRegression(C=1e5)
    #if method == 'linear':
    else:
        regr = sklearn.linear_model.LinearRegression()

    training_data_x = training_data.drop('Class', axis=1)
    training_data_y = training_data['Class']

    regr.fit(training_data_x, training_data_y)

    return regr


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
    return segment_groups.aggregate(preictal_ratio)
