"""Module for running the classification pipeline in python"""
import glob
import os.path
import datetime
import pickle

import sklearn
from sklearn import cross_validation
import pandas as pd

import correlation_convertion as corr_conv
import dataset
import seizure_modeling


def run_batch_classification(feature_folder_root="../../data/cross_correlation", **kwargs):
    all_scores = []
    for subject in ("Dog_1", "Dog_2", "Dog_3",
                      "Dog_4", "Dog_5", "Patient_1",
                      "Patient_2"):
        segment_scores = run_classification(os.path.join(feature_folder_root, subject), **kwargs)
        all_scores.append(segment_scores)

    #df_scores = pd.concat(all_scores, axis=0)
    #df_scores.sort()
    #timestamp = datetime.datetime.now().replace(microsecond=0)
    #submission_file = "submission_{}.csv".format(timestamp)
    #submission_path = os.path.join(feature_folder_root, submission_file)

    #df_scores.to_csv(submission_path, index_label='clip')


def get_latest_model(feature_folder, model_pattern="model*.pickle"):
    model_glob = os.path.join(feature_folder, model_pattern)
    files = glob.glob(model_glob)
    times = [(os.path.getctime(model_file),model_file)
                               for model_file in files]
    if times:
        ctime, latest_model = max(times)
        return latest_model
    else:
        return None


def write_scores(feature_folder, test_data, model, timestamp=None):
    """
    Writes a score file to *feature_folder*, using the scores given by
    *model* on *test_data*. If *timestamp* is supplied, it will be
    used for the file, otherwise a new timestamp will be generated.
    Returns the data frame of the calculated segment scores.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().replace(microsecond=0)

    segment_scores = seizure_modeling.assign_segment_scores(test_data, model)
    score_file = "classification_{}.csv".format(timestamp)
    score_path = os.path.join(feature_folder, score_file)
    segment_scores.to_csv(score_path, index_label='file')
    return segment_scores



def run_classification(feature_folder, rebuild_data=False,
                       training_ratio=.8, rebuild_model=False, model_file=None,
                       do_downsample=False, method="logistic", do_segment_split=False,
                       processes=4, csv_directory=None):
    print("Running classification on folder {}".format(feature_folder))
    interictal, preictal, unlabeled = corr_conv.load_data_frames(feature_folder,
                                                                 rebuild_data=rebuild_data,
                                                                 processes=processes)

    training_data, test_data = dataset.split_experiment_data(interictal,
                                                             preictal,
                                                             training_ratio=training_ratio,
                                                             do_downsample=do_downsample,
                                                             do_segment_split=do_segment_split)

    if model_file is None or not rebuild_model:
        model_file = get_latest_model(feature_folder)
        if model_file is None:
            rebuild_model = True
        else:
            with open(model_file, 'rb') as fp:
                model = pickle.load(fp, encoding='bytes')

    timestamp = datetime.datetime.now().replace(microsecond=0)
    if rebuild_model:
        model = seizure_modeling.train_model(training_data,
                                             method=method,
                                             do_segment_split=do_segment_split,
                                             processes=processes)
        if model_file is None:
            #Create a new filename based on the model method and the
            #date
            model_basename = "model_{}_{}.pickle".format(method, timestamp)
            model_file = os.path.join(feature_folder, model_basename)
        with open(model_file, 'wb') as fp:
            pickle.dump(model, fp)

    scores = write_scores(feature_folder, unlabeled, model, timestamp=timestamp)
    return scores

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="""Script for running the classification pipeline""")
    parser.add_argument("feature_folder_root", help="""The folder containing the features collected in subject subfolders""", default="../../data/cross_correlation")
    parser.add_argument("--rebuild-data", action='store_true', help="Should the dataframes be re-read from the csv feature files", dest='rebuild_data')
    parser.add_argument("--training-ratio", type=float, default=0.8, help="What ratio of the data should be used for training", dest='training_ratio')
    parser.add_argument("--rebuild-model", action='store_true', help="Should the model be rebuild, or should a cached version (if available) be used.", dest='rebuild_model')
    parser.add_argument("--no-downsample",
                        action='store_false',
                        default=True,
                        help="Disable downsampling of the majority class",
                        dest='do_downsample')
    parser.add_argument("--no-segment-split",
                        action='store_false',
                        help="Disable splitting data by segment.",
                        dest='do_segment_split',
                        default=True)
    parser.add_argument("--method", help="What model to use for learning", dest='method', choices=['logistic'], default='logistic')
    parser.add_argument("--processes", help="How many processes should be used for parellelized work.", dest='processes', default=4, type=int)
    parser.add_argument("--csv-directory", help="Which directory the classification CSV files should be written to.", dest='csv_directory')

    args = parser.parse_args()
    print("Starting training with the following arguments: {}".format(vars(args)))
    run_batch_classification(**vars(args))
