from __future__ import print_function

try:
    import cPickle as pickle
except ImportError:
    import pickle
import glob
import os.path
from classification_pipeline import get_latest_model, write_scores
from time import strftime, localtime
import random
import sys
import pandas as pd
import numpy as np
import multiprocessing
from functools import partial
# multiprocessing.set_start_method('spawn')
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import *
from sklearn.externals import joblib


# TODO: Provide singleton random generator

def run_batch_classification(feature_folder_root="../../data/wavelets",
                             rebuild_data=False, training_ratio=1.0,
                             rebuild_model=False, do_downsample=True,
                             method="svm"):

    for subject in ("Dog_1", "Dog_2", "Dog_3",
                    "Dog_4", "Dog_5", "Patient_1",
                    "Patient_2"):
        run_classification(
            os.path.join(feature_folder_root, subject),
            rebuild_data=rebuild_data,
            training_ratio=training_ratio, rebuild_model=rebuild_model,
            do_down_sample=do_downsample, method=method,
            do_segment_split=do_segment_split)

# def get_latest_model(feature_folder, model_pattern="model*.pickle"):
#     model_glob = os.path.join(feature_folder, model_pattern)
#     files = glob.glob(model_glob)
#     times = [(os.path.getctime(model_file),model_file)
#                                for model_file in files]
#     if times:
#         ctime, latest_model = max(times)
#         return latest_model
#     else:
#         return None

def run_classification(feature_folder, rebuild_data=False, training_ratio=1.0,
                      rebuild_model=False, model_file=None, do_downsample=True,
                      method="svm", do_segment_split=False, processes=4, seed=None, **kwargs):
    print("Running classification on folder {}".format(feature_folder))

    interictal, preictal, unlabeled = load_data_frames(
        feature_folder, rebuild_data=rebuild_data, processes=processes)


    # training_data, _ = split_experiment_data(
    #     interictal,preictal, training_ratio=training_ratio,
    #     do_downsample=do_downsample)

    # if model_file is None or not rebuild_model:
    #     model_file = get_latest_model(feature_folder)
    #     if model_file is None:
    #         rebuild_model = True
    #     else:
    #         with open(model_file, 'rb') as fp:
    #             model = pickle.load(fp, encoding='bytes')

    timestamp = strftime("%m-%d-%Y-%H.%M.%S", localtime())
    if rebuild_model:
        model = find_best_model(
            feature_folder, rebuild_data=False, training_ratio=0.8,
            scores=None, jobs=processes, do_downsample=True,
            method=method, seed=seed, **kwargs)

        if model_file is None:
            #Create a new filename based on the model method and the
            #date
            model_basename = "model_{}_{}.pickle".format(method, timestamp)
            model_file = os.path.join(feature_folder, model_basename)
        with open(model_file, 'wb') as fp:
            pickle.dump(model, fp)

    scores = write_scores(feature_folder, unlabeled, model, timestamp=timestamp)
    return model, scores

def extend_data_with_sliding_frames(source_array, frame_length=12):

    n_rows = source_array.shape[0]
    window_size = source_array.shape[1]

    #Number of frames that we can generate
    n_sliding_frames = n_rows-(frame_length-1)
    #The column size of our new frames
    frame_size = window_size*frame_length

    dest_array = np.zeros((n_sliding_frames, frame_size), dtype=np.float64)

    for i in range(0,n_sliding_frames):
        dest_array[i] = source_array[i:i+frame_length].reshape(1,frame_size)

    return dest_array

def load_csv(filename, frame_length=12, sliding_frames=True):

    #Read the csvfile with pandas and extract the values into an numpy array
    from_file_array = pd.read_table(filename, sep=',', dtype=np.float64, header=None).values

    # Assert that the csvfiles contain frames consisting 12 windows.
    assert_msg = 'file: "{}" does not have a column count divisible by 12 since it is: {}.'
    assert (from_file_array.shape[1] % 12) == 0, assert_msg.format(filename,from_file_array.shape[1])

    #Number of windows in the csv frame
    window_size = from_file_array.shape[1] / 12
    #Number of rows in the csv file
    n_rows = from_file_array.shape[0]*12

    reshaped_array = from_file_array.reshape(n_rows,window_size)

    #Extract this function out into its own file and use it also with the cross correlation frames
    if sliding_frames:

        return pd.DataFrame(data=extend_data_with_sliding_frames(reshaped_array, frame_length))

    else:
        n_frames = reshaped_array.shape[0]/frame_length
        frame_size = window_size*frame_length
        return pd.DataFrame(data=reshaped_array.reshape(n_frames,frame_size))

def load_wavelet_files(feature_folder,
                       class_name,
                       file_pattern="extract_features_for_segment.csv",
                       rebuild_data=False,
                       processes=1,
                       frame_length=12, sliding_frames=True):
    cache_file = os.path.join(
        feature_folder, '{}_frame_length_{}_cache.pickle'.format(class_name, frame_length))

    if rebuild_data or not os.path.exists(cache_file):
        print("Rebuilding {} data".format(class_name))
        full_pattern = "*{}*{}".format(class_name, file_pattern)
        glob_pattern = os.path.join(feature_folder, full_pattern)
        files = glob.glob(glob_pattern)
        segment_names = [os.path.basename(filename) for filename in files]
        if processes > 1:
            print("Reading files in parallel")
            pool = multiprocessing.Pool(processes)
            try:
                partial_load_csv = partial(load_csv, frame_length=frame_length, sliding_frames=sliding_frames)
                segment_frames = pool.map(partial_load_csv, files)
            finally:
                pool.close()
        else:
            print("Reading files serially")
            segment_frames = [load_csv(filename, frame_length=frame_length) for filename in files]

        complete_frame = pd.concat(segment_frames,
                                   names=('segment', 'frame'),
                                   keys=segment_names)
        # Sorting for optimization
        complete_frame.sortlevel(inplace=True)

        if np.count_nonzero(np.isnan(complete_frame)) != 0:
            print("WARNING: NaN values found, using interpolation",
                  file=sys.stderr)
            complete_frame = complete_frame.interpolate(method='linear')

        complete_frame.to_pickle(cache_file)
    else:
        complete_frame = pd.read_pickle(cache_file)
    return complete_frame


def load_data_frames(feature_folder, rebuild_data=False,
                     processes=4,
                     file_pattern="extract_features_for_segment.csv",
                     frame_length=12):

    preictal = load_wavelet_files(feature_folder,
                                  class_name="preictal",
                                  file_pattern=file_pattern,
                                  rebuild_data=rebuild_data,
                                  processes=processes,
                                  frame_length=frame_length)
    preictal['Preictal'] = 1

    interictal = load_wavelet_files(feature_folder,
                                    class_name="interictal",
                                    file_pattern=file_pattern,
                                    rebuild_data=rebuild_data,
                                    processes=processes,
                                    frame_length=frame_length)
    interictal['Preictal'] = 0

    test = load_wavelet_files(feature_folder,
                              class_name="test",
                              file_pattern=file_pattern,
                              rebuild_data=rebuild_data,
                              processes=processes,
                              frame_length=frame_length,
                              sliding_frames=False)

    return interictal, preictal, test

def split_experiment_data(train_interictal, train_preictal, training_ratio=0.8,
                          do_downsample=True, downsample_ratio=1.0, seed=None):
    # Figure how many positive and negative samples we have in the complete
    # training set.
    interictal_samples = train_interictal.shape[0]
    preictal_samples = train_preictal.shape[0]

    if preictal_samples > interictal_samples:
       print("WARNING: preictal samples: %d, interictal samples: %d" %
             (preictal_samples, interictal_samples), file=sys.stderr)

    if do_downsample:

        # Get approximatelly downsample_ratio * preictal_samples from the
        # interictal_samples sframe

        desired_interictal = downsample_ratio * preictal_samples
        # interictal_ratio = desired_interictal / float(interictal_samples)

        # TODO: The sampling of the interictal data here should also be done
        # in a stratified manner, to avoid having many samples from the same
        # segment
        if seed is None:
            # Seed needs to be small or sample complains, hence the division
            seed = int(random.randrange(0, 4294967295)) #max int
            np.random.seed(seed)
            # TODO: try: no divide; catch: divide with e8 if error

        downsampled_interictal, _ = random_split(train_interictal,
                                                 desired_rows=desired_interictal, seed=seed)

        downsampled_complete = downsampled_interictal.append(train_preictal)
        # OK to re-use seed?
        print()
        print("Original interictal samples: %d" % interictal_samples)
        print("Original preictal samples: %d" % preictal_samples)
        print("Interictal samples after downsampling: %d" % downsampled_interictal.shape[0])
        return random_split(downsampled_complete, ratio=training_ratio, seed=seed)
    else:
        complete = train_interictal.append(train_preictal)
        print()
        print("Original interictal samples: %d" % interictal_samples)
        print("Original preictal samples: %d" % preictal_samples)
        print("Combined samples: %d" % complete.shape[0])
        return random_split(complete, ratio=training_ratio, seed=seed)


def random_split(dataframe, ratio=None, desired_rows=None, seed=None):

    assert ((ratio is not None) or (desired_rows is not None)), (
        "You have to provide either number of rows or ratio")
    if seed is None:
        # Seed needs to be small or sample complains, hence the division
        seed = int(random.randrange(0, 4294967295))
        np.random.seed(seed)

    if ratio is not None:
        if ratio >= 1.0:
            return dataframe
        msk = np.random.rand(len(dataframe)) < ratio
        return dataframe[msk], dataframe[~msk] #train, test
    else:
        if dataframe.shape[0] <= desired_rows:
            return dataframe
        rows = np.random.choice(dataframe.index.values,
                                desired_rows, replace=False)
        return dataframe.ix[rows], dataframe.drop(rows)

def create_model(method, **kwargs):

    if method == 'svm':
        clf = SVC(kwargs, probability=True, class_weight='auto')
    elif method == 'sgd':
        clf = SGDClassifier(**kwargs)
    elif method == 'random-forest':
        clf = RandomForestClassifier(**kwargs)
    else:
        raise NotImplementedError("Method %s is not supported" % method)

    return clf

def parameters_for_method(method):
    if method == 'svm':
        return [{'kernel': ['rbf'], 'gamma': [0, 1e-1, 1e-2, 1e-3],
                 'C': [10, 100, 1000]}]
    elif method == 'sgd':
        return [{'loss' : ['modified_huber', 'log'],
                'penalty' : ['l1', 'l2', 'elasticnet'],
                'alpha' : [0.0001, 0.001, 0.01, 0.1]}]


def run_cross_validation(feature_folder, rebuild_data=False, training_ratio=.8,
                         jobs=4, do_downsample=True, method="svm", seed=None, **kwargs):

    print("Running cross validation on folder {}".format(feature_folder))

    print("Loading data")
    interictal, preictal, test = load_data_frames(feature_folder, rebuild_data=rebuild_data)

    print("Splitting/downsampling data")
    complete_data, _ = split_experiment_data(
        interictal, preictal, training_ratio=1.0,
        do_downsample=do_downsample, seed=seed)

    model = create_model(method=method, **kwargs)

    target = "Preictal"
    y = complete_data.pop(target).to_dense()
    X = complete_data

    # TODO: To shuffle or not to shuffle?
    # See: http://scikit-learn.org/stable/modules/cross_validation.html#a-note-on-shuffling
    # if do_downsample:
    #     cv = KFold(complete_data.shape[0], n_folds=10)
    # else:
    cv = StratifiedKFold(y, n_folds=10)

    print("Running cross validation")
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=jobs)

    def print_scores(scores):
        print("ROC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print_scores(scores)

    return model, scores

def find_best_model(feature_folder, rebuild_data=False, training_ratio=0.8,
                    scores=None, jobs=4, do_downsample=True,
                    method="svm", seed=None, **kwargs):

    print("Running model search on folder {}".format(feature_folder))

    print("Loading data")
    interictal, preictal, _ = load_data_frames(
        feature_folder, rebuild_data=rebuild_data)

    print("Splitting/downsampling data")
    complete_data, _ = split_experiment_data(
        interictal, preictal, training_ratio=1.0,
        do_downsample=do_downsample, seed=seed)

    model = create_model(method=method, **kwargs)
    tuned_parameters = parameters_for_method(method)

    target = "Preictal"
    y = complete_data.pop(target).to_dense()
    X = complete_data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=training_ratio, random_state=seed)

    # TODO: To shuffle or not to shuffle?
    # See: http://scikit-learn.org/stable/modules/cross_validation.html#a-note-on-shuffling
    # if do_downsample:
    #     cv = KFold(complete_data.shape[0], n_folds=10)
    # else:
    cv = StratifiedKFold(y_train, n_folds=10)

    if scores is None:
        scores = [make_scorer(roc_auc_score, average='weighted')]

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            model, tuned_parameters, cv=cv, scoring=score, n_jobs=jobs)
        clf.fit(X_train, y_train)

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
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        print_cm(confusion_matrix(y_true, y_pred),
                 labels=['Interictal', 'Preictal'])


    timestamp = strftime("%m-%d-%Y-%H.%M.%S", localtime())
    # Create a new filename based on the model method and the
    # date
    model_basename = "model_{}_{}.pickle".format(method, timestamp)
    model_file = os.path.join(feature_folder, model_basename)
    try:
        # joblib.dump(clf, model_file)
        with open(model_file, 'wb') as fp:
            pickle.dump(model, fp)
    except TypeError:
        print("Could not save model file", file=sys.stderr)

    return clf



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



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="""Script for running the classification pipeline""")
    parser.add_argument("feature_folder_root", help="""The folder containing the features collected in subject subfolders""", default="../../data/cross_correlation")
    parser.add_argument("--rebuild-data", action='store_true', help="Should the dataframes be re-read from the csv feature files", dest='rebuild_data')
    parser.add_argument("--training-ratio", type=float, default=0.8, help="What ratio of the data should be used for training", dest='training_ratio')
    parser.add_argument("--rebuild-model", action='store_true', help="Should the model be rebuild, or should a cached version (if available) be used.", dest='rebuild_model')
    parser.add_argument("--do-downsample", action='store_true', help="should class imbalance be solved by downsampling the majority class", dest='do_downsample')
    parser.add_argument("--do-segment-split", help="Should the training data sampling be done on a per segment basis.", dest='method')
    parser.add_argument("--method", help="What model to use for learning", dest='method')

    args = parser.parse_args()
    run_batch_classification(**vars(args))
