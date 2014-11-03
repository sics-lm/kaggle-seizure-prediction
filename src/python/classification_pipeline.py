"""Module for running the classification pipeline in python"""
import glob
import os
import os.path
import datetime
import pickle
import re
import logging
import sys

import pandas as pd

import correlation_convertion
import wavelet_classification
import dataset
import seizure_modeling
import fileutils

def run_batch_classification(feature_folders, **kwargs):
    """
    Runs the batch classificatio on the feature folders.
    Args:
        feature_folders: Should be a list of folders containing feature files or folders containing the canonical subject folders ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']. If the folder contains the subject folders, it will be replaced by them in the list of feature folders. For example feature_folders = ['../../data/cross_correlations', '../../data/maximal_xcorr/Dog_1'] would be expanded to feature_folders = ['../../data/cross_correlations/Dog_1', '../../data/cross_correlations/Dog_2', '../../data/cross_correlations/Dog_3', '../../data/cross_correlations/Dog_4', '../../data/cross_correlations/Dog_5', '../../data/cross_correlations/Patient_1', '../../data/cross_correlations/Patient_2']', '../../data/maximal_xcorr/Dog_1'].
    """
    feature_folders = fileutils.expand_folders(feature_folders)
    all_scores = []
    for subject_folder in feature_folders:
        segment_scores = run_classification(subject_folder, **kwargs)
        all_scores.append(segment_scores)

    # df_scores = pd.concat(all_scores, axis=0)
    # df_scores.sort()
    # timestamp = datetime.datetime.now().replace(microsecond=0)
    # submission_file = "submission_{}.csv".format(timestamp)
    # submission_path = os.path.join(feature_folder_root, submission_file)

    # df_scores.to_csv(submission_path, index_label='clip')


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



def run_classification(feature_folder,
                       rebuild_data=False,
                       training_ratio=.8,
                       rebuild_model=False,
                       model_file=None,
                       do_downsample=False,
                       downsample_ratio=2.0,
                       method="logistic",
                       do_segment_split=False,
                       processes=4,
                       csv_directory=None,
                       feature_type='cross-correlations',
                       frame_length=1):
    logging.info("Running classification on folder {}".format(feature_folder))
    if feature_type == 'wavelets':
        interictal, preictal, unlabeled = wavelet_classification.load_data_frames(feature_folder,
                                                                                   rebuild_data=rebuild_data,
                                                                                   processes=processes, frame_length=frame_length)
    elif feature_type == 'cross-correlations' or feature_type == 'xcorr':
        interictal, preictal, unlabeled = correlation_convertion.load_data_frames(feature_folder,
                                                                                  rebuild_data=rebuild_data,
                                                                                  processes=processes, frame_length=frame_length)

    if model_file is None or not rebuild_model:
        model_file = get_latest_model(feature_folder)
        if model_file is None:
            rebuild_model = True
        else:
            with open(model_file, 'rb') as fp:
                model = pickle.load(fp, encoding='bytes')

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    if rebuild_model:
        model = seizure_modeling.train_model(interictal, preictal,
                                             method=method,
                                             do_downsample=do_downsample,
                                             downsample_ratio=downsample_ratio,
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
    logging.info("Finnished with classification on folder {}".format(feature_folder))

    return scores


def setup_logging(args):
    log_dir = args['log_dir']
    del args['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    log_file = "classification_log-method:{}-frame_length:{}-time:{}.txt".format(args['method'], args['frame_length'], timestamp)
    log_path = os.path.join(log_dir, log_file)


    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)

    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(logFormatter)

    logging.basicConfig(level='INFO', handlers=(fileHandler, std_handler))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="""Script for running the classification pipeline""")

    parser.add_argument("feature_folders",
                        help="""The folders containing the features""",
                        nargs='+')
    parser.add_argument("-t", "--feature-type",
                        help="""The type of the features""",
                        choices=["wavelets", "cross-correlations", "xcorr"],
                        required=True,
                        dest='feature_type')
    parser.add_argument("--rebuild-data",
                        action='store_true',
                        help="Should the dataframes be re-read from the csv feature files",
                        dest='rebuild_data')
    parser.add_argument("--training-ratio",
                        type=float,
                        default=0.8,
                        help="What ratio of the data should be used for training",
                        dest='training_ratio')
    parser.add_argument("--rebuild-model",
                        action='store_true',
                        help="Should the model be rebuild, or should a cached version (if available) be used.",
                        dest='rebuild_model')
    parser.add_argument("--no-downsample",
                        action='store_false',
                        default=True,
                        help="Disable downsampling of the majority class",
                        dest='do_downsample')
    parser.add_argument("--downsample-ratio",
                        default=2.0,
                        type=float,
                        help="The raio of majority class to minority class after downsampling.",
                        dest='downsample_ratio')
    parser.add_argument("--no-segment-split",
                        action='store_false',
                        help="Disable splitting data by segment.",
                        dest='do_segment_split',
                        default=True)
    parser.add_argument("--method",
                        help="What method to use for learning",
                        dest='method', choices=['logistic', 'svm', 'sgd'],
                        default='logistic')
    parser.add_argument("--processes",
                        help="How many processes should be used for parellelized work.",
                        dest='processes',
                        default=4,
                        type=int)
    parser.add_argument("--csv-directory",
                        help="Which directory the classification CSV files should be written to.",
                        dest='csv_directory')
    parser.add_argument("--frame-length",
                        help="The size in windows each frame (feature vector) should be.",
                        dest='frame_length', default=1, type=int)
    parser.add_argument("--log-dir",
                        help="Directory for writing classification log files.",
                        default='../../classification_logs',
                        dest='log_dir')


    args_dict = vars(parser.parse_args())
    ## Setup loging stuff, this removes 'log_dir' from the dictionary
    setup_logging(args_dict)

    logging.info("Starting training with the following arguments: {}".format(args_dict))

    run_batch_classification(**args_dict)
