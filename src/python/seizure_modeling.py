"""Module for doing the training of the models."""

import sklearn
import sklearn.linear_model
import sklearn.cross_validation

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


def assign_segment_scores(test_data, regr):
    predictions = regr.predict(test_data)
    test_data['Guess'] = predictions
