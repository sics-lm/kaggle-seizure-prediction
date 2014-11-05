"""Module for running the classification pipeline in python"""
import glob
import os
import os.path
import datetime
import pickle
import logging
import sys

import correlation_convertion
import wavelet_classification
import dataset
import seizure_modeling
import fileutils
import submissions


def run_batch_classification(feature_folders, timestamp, submission_file=None, **kwargs):
    """Runs the batch classificatio on the feature folders.
    Args:

        feature_folders: Should be a list of folders containing feature
                         files or folders containing the canonical subject folders
                         {'Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1',
                        'Patient_2'}. If the folder contains any of the subject
                         folders, it will be replaced by them in the list of feature
                         folders.
        submission_file: If this argument is a path, the classification scores
                         will be written to a csv file with that path.
    Returns:
        None.
    """
    feature_folders = fileutils.expand_folders(feature_folders)
    all_scores = []
    for subject_folder in feature_folders:
        segment_scores = run_classification(subject_folder, **kwargs)
        score_dict = segment_scores.to_dict()['preictal']
        all_scores.append(score_dict)

    if submission_file is None:
        name_components = ["submission"]
        name_components.append(kwargs['feature_type'])
        name_components.append(kwargs['method'])
        if kwargs['do_standardize']:
            name_components.append("standardized")
        name_components.append(str(timestamp))
        filename = '_'.join(name_components) + '.csv'
        submission_file = os.path.join('..', '..', 'submissions', filename)

    logging.info("Saving submission scores to {}".format(submission_file))
    with open(submission_file, 'w') as fp:
        submissions.write_scores(all_scores, output=fp)


def write_scores(csv_directory, test_data, model, timestamp=None):
    """
    Writes the model prediction scores for the segments of *test_data* to a csv file.
    Args:
        csv_directory: The director to where the classification scores will be written.
        test_data: The dataframe holding the test data
        model: The model to use for predicting the preictal probability of the test data.
        timestamp: If this argument is given, it will be used for naming the
                   classification file. If it's not given, the current time
                   will be used as a time stamp for the file.
    Returns:
        A dataframe containing the segment preictal probabilities.
    """

    if timestamp is None:
        timestamp = datetime.datetime.now().replace(microsecond=0)

    segment_scores = seizure_modeling.assign_segment_scores(test_data, model)
    score_file = "classification_{}.csv".format(timestamp)
    score_path = os.path.join(csv_directory, score_file)
    logging.info("Writing classification scores to {}.".format(score_path))
    segment_scores.to_csv(score_path, index_label='file')
    return segment_scores


def get_latest_model(feature_folder, method, model_pattern="model*{method}*.pickle"):
    model_glob = os.path.join(feature_folder, model_pattern.format(method=method))
    files = glob.glob(model_glob)
    times = [(os.path.getctime(model_file), model_file)
             for model_file in files]
    if times:
        _, latest_model = max(times)
        print("Latest model is:", latest_model)
        with open(latest_model, 'rb') as fp:
            logging.info("Loading classifier from {}.".format(latest_model))
            model = pickle.load(fp, encoding='bytes')
            return model
    else:
        return None


def run_classification(feature_folder,
                       rebuild_data=False,
                       training_ratio=.8,
                       rebuild_model=False,
                       model_file=None,
                       do_downsample=False,
                       downsample_ratio=2.0,
                       do_standardize=False,
                       method="logistic",
                       do_segment_split=False,
                       processes=4,
                       csv_directory=None,
                       feature_type='cross-correlations',
                       frame_length=1,
                       do_refit=True):
    logging.info("Running classification on folder {}".format(feature_folder))
    if feature_type == 'wavelets':
        interictal, preictal, unlabeled = wavelet_classification.load_data_frames(feature_folder,
                                                                                  rebuild_data=rebuild_data,
                                                                                  processes=processes, frame_length=frame_length)
    elif feature_type == 'cross-correlations' or feature_type == 'xcorr':
        interictal, preictal, unlabeled = correlation_convertion.load_data_frames(feature_folder,
                                                                                  rebuild_data=rebuild_data,
                                                                                  processes=processes, frame_length=frame_length)
    if do_standardize:
        logging.info("Standardizing variables.")
        interictal, preictal, unlabeled = dataset.scale(interictal,
                                                        preictal,
                                                        unlabeled,
                                                        inplace=True)

    if model_file is None and not rebuild_model:
        model = get_latest_model(feature_folder, method)
        if model is None:
            rebuild_model = True

    timestamp = None
    if rebuild_model:
        timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
        model = seizure_modeling.train_model(interictal, preictal,
                                             method=method,
                                             do_downsample=do_downsample,
                                             downsample_ratio=downsample_ratio,
                                             do_segment_split=do_segment_split,
                                             training_ratio=training_ratio,
                                             processes=processes)
        if model_file is None:
            #Create a new filename based on the model method and the
            #date
            model_basename = "model_{}_{}.pickle".format(method, timestamp)
            model_file = os.path.join(feature_folder, model_basename)
        with open(model_file, 'wb') as fp:
            pickle.dump(model, fp)

    if do_refit:
        logging.info("Refitting model with held-out data.")
        model = seizure_modeling.refit_model(interictal,
                                             preictal,
                                             model,
                                             do_downsample=do_downsample,
                                             downsample_ratio=downsample_ratio,
                                             do_segment_split=do_segment_split)

    if csv_directory is None:
        csv_directory = feature_folder
    scores = write_scores(csv_directory, unlabeled, model, timestamp=timestamp)
    logging.info("Finnished with classification on folder {}".format(feature_folder))

    return scores


def setup_logging(timestamp, args):
    log_dir = args['log_dir']
    del args['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = "classification_log-method:{}-frame_length:{}-time:{}.txt".format(args['method'], args['frame_length'], timestamp)
    log_path = os.path.join(log_dir, log_file)

    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(log_formatter)

    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)

    logging.basicConfig(level='INFO', handlers=(file_handler, std_handler))


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
    parser.add_argument("--standardize",
                        action='store_true',
                        help="Standardize the variables",
                        dest='do_standardize',
                        default=False)
    parser.add_argument("--no-refit",
                        action='store_false',
                        default=True,
                        help="Don't refit the selected model with the held-out data used to produce accurace scores",
                        dest='do_refit')
    parser.add_argument("--no-segment-split",
                        action='store_false',
                        help="Disable splitting data by segment.",
                        dest='do_segment_split',
                        default=True)
    parser.add_argument("--method",
                        help="What method to use for learning",
                        dest='method',
                        choices=['logistic',
                                 'svm',
                                 'sgd',
                                 'random-forest'],
                        default='logistic')
    parser.add_argument("--processes",
                        help="How many processes should be used for parellelized work.",
                        dest='processes',
                        default=4,
                        type=int)
    parser.add_argument("--csv-directory",
                        help="Which directory the classification CSV files should be written to.",
                        dest='csv_directory')
    parser.add_argument("--submission-file",
                        help="""If this argument is supplied, a submissions file
                        with the scores for the the test segments will be produced""",
                        dest='submission_file')
    parser.add_argument("--frame-length",
                        help="The size in windows each frame (feature vector) should be.",
                        dest='frame_length', default=1, type=int)
    parser.add_argument("--log-dir",
                        help="Directory for writing classification log files.",
                        default='../../classification_logs',
                        dest='log_dir')


    args_dict = vars(parser.parse_args())

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    ## Setup loging stuff, this removes 'log_dir' from the dictionary
    setup_logging(timestamp, args_dict)

    logging.info("Starting training with the following arguments: {}".format(args_dict))

    run_batch_classification(timestamp=timestamp, **args_dict)
