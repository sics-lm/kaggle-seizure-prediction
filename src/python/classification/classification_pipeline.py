"""Module for running the classification pipeline in python"""
from __future__ import absolute_import
import glob
import os
import os.path
import datetime
import pickle
import logging
import sys

import numpy as np

from . import submissions, seizure_modeling

from ..datasets import dataset, fileutils, features_combined, correlation_convertion, wavelet_classification


def run_batch_classification(feature_folders,
                             timestamp,
                             file_components=None,
                             optional_file_components=None,
                             submission_file=None,
                             frame_length=1,
                             sliding_frames=False,
                             rebuild_data=False,
                             feature_type='hills',
                             processes=1,
                             csv_directory=None,
                             do_downsample=True,
                             downsample_ratio=2.0,
                             do_standardize=False,
                             do_segment_split=True,
                             do_pca=False,
                             random_state=None,
                             **kwargs):
    """
    Runs the batch classification on the feature folders.

    :param feature_folders: Should be a list of folders containing feature files or folders containing the canonical
    subject folders {'Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2'}.
    If the folder contains any of the subject folders, it will be replaced by them in the list of feature folders.
    :param timestamp: The provided timestamp for this classification
    :param file_components: ???
    :param optional_file_components: ???
    :param submission_file: If this argument is a path, the classification scores will be written to a csv file with
    that path.
    :param frame_length: The length of each frame, in number of windows
    :param sliding_frames: If True, sliding windows will be used
    :param rebuild_data: If True, the cache files will be rebuilt from the csv data.
    :param feature_type: The name of the feature. Valid values are 'cross-correlation', 'hills' and 'wavelets'
    :param processes: The number of processes to use in the grid search
    :param csv_directory: Where to place the resulting classification files
    :param do_downsample: If true downsampling of the interictal features will be performed
    :param downsample_ratio: The ratio of interictal to preictal samples after downsampling.
    :param do_standardize: If True, the features will be standarized before classification
    :param do_segment_split: If True, the cross-validation will be performed by splitting on whole segments
    :param do_pca: If True, the the PCA feature reduction will be performed on the data.
    :param random_state: Seed
    :param kwargs: Incoming kwargs that will be passed on to the classifiers
    :return: None. Creates the output classification and submission files in ../submissions.
    """

    all_scores = []
    for feature_dict in load_features(feature_folders,
                                      feature_type=feature_type,
                                      frame_length=frame_length,
                                      sliding_frames=sliding_frames,
                                      rebuild_data=rebuild_data,
                                      processes=processes,
                                      do_downsample=do_downsample,
                                      downsample_ratio=downsample_ratio,
                                      do_standardize=do_standardize,
                                      do_segment_split=do_segment_split,
                                      do_pca=do_pca,
                                      random_state=random_state):
        kwargs.update(feature_dict)  # Adds the content of feature dict to the keywords for run_classification
        segment_scores = run_classification(processes=processes,
                                            csv_directory=csv_directory,
                                            file_components=file_components,
                                            optional_file_components=optional_file_components,
                                            random_state=random_state,
                                            **kwargs)
        score_dict = segment_scores.to_dict()['Preictal']
        all_scores.append(score_dict)

    if submission_file is None or os.path.isdir(submission_file):
        #We need to generate a new submission filename
        if file_components is None:
            file_components = [feature_type,
                               kwargs['method'],
                               'frame_length_{}'.format(frame_length)]
        if optional_file_components is None:
            optional_file_components = dict(standardized=do_standardize,
                                            sliding_frames=sliding_frames)

        filename = fileutils.generate_filename('submission',
                                               '.csv',
                                               file_components,
                                               optional_file_components,
                                               timestamp=timestamp)
        #Let the submission path take precedence if it's a folder
        if submission_file is not None:
            submission_file = os.path.join(submission_file, filename)
        elif csv_directory is not None:
            submission_file = os.path.join(csv_directory, filename)
        else:
            submission_file = os.path.join('..', '..', 'submissions', filename)

    logging.info("Saving submission scores to {}".format(submission_file))
    with open(submission_file, 'w') as fp:
        submissions.write_scores(all_scores, output=fp, do_normalize=True)


def load_features(feature_folders,
                  feature_type='cross-correlations',
                  frame_length=1,
                  sliding_frames=False,
                  rebuild_data=False,
                  processes=1,
                  do_downsample=False,
                  downsample_ratio=2.0,
                  do_standardize=False,
                  do_segment_split=True,
                  do_pca=False,
                  random_state=None):
    """
    Loads the features from the list of paths *feature_folder*. Returns an iterator of dictionaries, where each
    dictionary has the keys 'subject_folder', 'interictal_data', ''preictal_data' and 'unlabeled_data'.

    :param feature_folders: A list of paths to folders containing features. The features in these folders will be
    combined into three data frames.
    :param feature_type: A string describing the type of features to use. If  'wavelets' is supplied, the feature files
    will be loaded as wavelets. If 'cross-correlations' or 'xcorr' is supplied, the features will be loaded as
    cross-correlation features. If 'combined' is supplied, the path of the feature folders will be used to determine
    which features it contains, and the results will be combined column-wise into longer feature vectors.
    :param frame_length: The desired frame length in windows of the features.
    :param sliding_frames: If True, the training feature frames will be generated by a sliding window, greatly
    increasing the number of generated frames.
    :param rebuild_data: If True, will rebuild the feature data cache files.
    :param processes: Number of processes to use in the loading of the data
    :param do_downsample:
    :param downsample_ratio:
    :param do_standardize:
    :param do_segment_split:
    :param do_pca:
    :param random_state:
    :return: A generator object which gives a dictionary with features for every call to next. The dictionary contains
    the keys 'subject_folder', 'interictal_data', 'preictal_data' and 'unlabeled_data'.
    """

    feature_folders = sorted(fileutils.expand_folders(feature_folders))

    if feature_type in ['wavelets', 'hills', 'cross-correlations', 'xcorr']:
        if feature_type == 'wavelets' or feature_type == 'hills':
            feature_module = wavelet_classification
        else:
            feature_module = correlation_convertion
        for feature_folder in feature_folders:
            interictal, preictal, unlabeled = feature_module.load_data_frames(feature_folder,
                                                                              rebuild_data=rebuild_data,
                                                                              processes=processes,
                                                                              frame_length=frame_length,
                                                                              sliding_frames=sliding_frames)
            interictal, preictal, unlabeled = preprocess_features(interictal, preictal, unlabeled,
                                                                  do_downsample=do_downsample,
                                                                  downsample_ratio=downsample_ratio,
                                                                  do_standardize=do_standardize,
                                                                  do_segment_split=do_segment_split,
                                                                  do_pca=do_pca,
                                                                  random_state=random_state)
            yield dict(interictal_data=interictal,
                       preictal_data=preictal,
                       unlabeled_data=unlabeled,
                       subject_folder=feature_folder)

    elif feature_type == 'combined':
        combined_folders = fileutils.group_folders(feature_folders)
        for subject, combo_folders in combined_folders.items():
            # We create an output folder which is based on the subject name
            subject_folder = os.path.join('..', '..', 'data', 'combined', subject)
            if not os.path.exists(subject_folder):
                os.makedirs(subject_folder)

            dataframes = dataset.load_data_frames(combo_folders,
                                                  load_function=features_combined.load,
                                                  find_features_function=fileutils.find_grouped_feature_files,
                                                  rebuild_data=rebuild_data,
                                                  processes=processes,
                                                  frame_length=frame_length,
                                                  sliding_frames=sliding_frames,
                                                  output_folder=subject_folder)
            interictal, preictal, unlabeled = dataframes
            yield dict(interictal_data=interictal,
                       preictal_data=preictal,
                       unlabeled_data=unlabeled,
                       subject_folder=subject_folder)
    else:
        raise NotImplementedError("No feature loading method implemented for feature type {}".format(feature_type))


def preprocess_features(interictal,
                        preictal,
                        test,
                        do_downsample=False,
                        downsample_ratio=2.0,
                        do_standardize=False,
                        do_segment_split=False,
                        do_pca=False,
                        random_state=None):
    """
    Performs pre-processing on the features.

    This can include downsampling the interictal data, standardizing the data and performing PCA feature reduction.
    """
    logging.info("Preprocessing features")
    if do_downsample:
        interictal = dataset.downsample(interictal,
                                        len(preictal)*downsample_ratio,
                                        do_segment_split=do_segment_split,
                                        random_state=random_state)
    if do_standardize:
        logging.info("Standardizing variables.")
        interictal, preictal, test = dataset.scale([interictal, preictal, test])
        logging.info("Shapes after standardization:")
        logging.info("Interictal: {}".format(interictal.shape))
        logging.info("Preictal: {}".format(preictal.shape))
        logging.info("Unlabeled: {}".format(test.shape))

    if do_pca:
        logging.info("Performing SVD decomposition of features")
        interictal, preictal, test = dataset.pca_transform([interictal, preictal, test])
        logging.info("Shapes after SVD decomposition:")
        logging.info("Interictal: {}".format(interictal.shape))
        logging.info("Preictal: {}".format(preictal.shape))
        logging.info("Unlabeled: {}".format(test.shape))

    return interictal, preictal, test


def run_classification(interictal_data,
                       preictal_data,
                       unlabeled_data,
                       subject_folder,
                       training_ratio=.8,
                       file_components=None,
                       optional_file_components=None,
                       model_file=None,
                       rebuild_model=False,
                       method="logistic",
                       do_segment_split=False,
                       processes=4,
                       csv_directory=None,
                       do_refit=True,
                       cv_verbosity=2,
                       model_params=None,
                       random_state=None,
                       no_crossvalidation=False):
    logging.info("Running classification on folder {}".format(subject_folder))

    if model_file is None and not rebuild_model:
        model = get_latest_model(subject_folder, method)
        if model is None:
            rebuild_model = True

    timestamp = None
    if rebuild_model:
        timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
        model = seizure_modeling.train_model(interictal_data,
                                             preictal_data,
                                             method=method,
                                             do_segment_split=do_segment_split,
                                             training_ratio=training_ratio,
                                             processes=processes,
                                             cv_verbosity=cv_verbosity,
                                             model_params=model_params,
                                             random_state=random_state,
                                             no_crossvalidation=no_crossvalidation)
        if model_file is None:
            # Create a new filename based on the model method and the
            # date
            if file_components is None:
                model_basename = "model_{}_{}.pickle".format(method, timestamp)
            else:
                model_basename = fileutils.generate_filename('model',
                                                             '.pickle',
                                                             components=file_components,
                                                             optional_components=optional_file_components,
                                                             timestamp=timestamp)
            model_file = os.path.join(subject_folder, model_basename)
        with open(model_file, 'wb') as fp:
            pickle.dump(model, fp)

    # If we don't use cross-validation we shouldn't refit
    if do_refit and not no_crossvalidation:
        logging.info("Refitting model with held-out data.")
        model = seizure_modeling.refit_model(interictal_data,
                                             preictal_data,
                                             model)

    if csv_directory is None:
        csv_directory = subject_folder
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)
    # TODO model might be referenced before assignment
    scores = write_scores(csv_directory, unlabeled_data, model, file_components=file_components, optional_file_components=optional_file_components, timestamp=timestamp)
    logging.info("Finnished with classification on folder {}".format(subject_folder))

    return scores


def write_scores(csv_directory, test_data, model, file_components=None, optional_file_components=None, timestamp=None):
    """
    Writes the model prediction scores for the segments of *test_data* to a csv file.

    :param csv_directory: The directory to where the classification scores will be written.
    :param test_data: The dataframe holding the test data
    :param model: The model to use for predicting the preictal probability of the test data.
    :param file_components: ???
    :param optional_file_components: ???
    :param timestamp: If this argument is provided, it will be used for naming the classification file.
    Otherwise the current time will be used as a time stamp for the file.
    :return: A dataframe containing the predicted segment preictal probabilities. The probabilities are also written
    to disk.
    """


    if timestamp is None:
        timestamp = datetime.datetime.now().replace(microsecond=0)

    segment_scores = seizure_modeling.assign_segment_scores(test_data, model)
    if file_components is None:
        score_file = "classification_{}.csv".format(timestamp)
    else:
        score_file = fileutils.generate_filename('classification','.csv', components=file_components,
                                                 optional_components=optional_file_components, timestamp=timestamp)
    score_path = os.path.join(csv_directory, score_file)
    logging.info("Writing classification scores to {}.".format(score_path))
    segment_scores.to_csv(score_path, index_label='file')
    return segment_scores


def get_latest_model(feature_folder, method, model_pattern="model*{method}*.pickle"):
    """
    Retrieve the latest cached model for specified feature folder
    """
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


def setup_logging(timestamp, file_components, optional_file_components, args):
    """
    Sets up the logger for the classification.
    :param timestamp: The timestamp to apply to the file
    :param args: a dictionary with the arguments which are used by the classifier. This dict will be modified,
                 removing items which shouldn't be sent to the classification function.
    :return: None
    """
    log_dir = args['log_dir']
    del args['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = fileutils.generate_filename('clog', '.txt',
                                           components=file_components,
                                           optional_components=optional_file_components,
                                           timestamp=timestamp)
    log_path = os.path.join(log_dir, log_file)

    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(log_formatter)

    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)

    logging.basicConfig(level='INFO', handlers=(file_handler, std_handler))


def fix_model_params(model_params_string):
    return eval(model_params_string)


def get_cli_args():
    """
    Returns the command line arguments.
    :return: A dictionary with the command line argument keys
    """
    import argparse
    parser = argparse.ArgumentParser(description="""Script for running the classification pipeline""")

    parser.add_argument("feature_folders",
                        help=("The folders containing the features. Multiple"
                              " paths can be specified. The given path will "
                              "be checked if it's a feature root folder, which"
                              " means it's a folder containing the canonical"
                              " subject directories. If that is the case,"
                              " it will be expanded into those subject folders."
                              " If it doesn't contain any canonical sujbect"
                              " folder, the argument is assumed to contain"
                              " feature files."),
                        nargs='+')
    parser.add_argument("-t", "--feature-type",
                        help=("The type of the features for the classification."
                              " 'cross-correlations' and 'xcorr' are synonymns."
                              "If the method is 'combined', the name of the "
                              "folder wil be used to decide which feature loader"
                              " to use. The folder must have the string "
                              "'wavelet' in it for the wavelet features and "
                              "the string 'corr' in it for cross correlation"
                              " features."),
                        choices=["wavelets",
                                 "cross-correlations",
                                 "xcorr",
                                 "hills",
                                 "combined"],
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
    parser.add_argument("--pca",
                        action='store_true',
                        help="Perform dimensionality reduction on the data using PCA",
                        dest='do_pca',
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
                                 'random-forest',
                                 'nearest-centroid',
                                 'knn',
                                 'bagging'],
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
                        dest='frame_length', default=12, type=int)
    parser.add_argument("--sliding-frames",
                        help=("If enabled, frames for the training-data will be"
                              " extracted by overlapping windows, greatly "
                              "increasing the number of frames."),
                        dest='sliding_frames',
                        default=False,
                        action='store_true')
    parser.add_argument("--log-dir",
                        help="Directory for writing classification log files.",
                        default='../../classification_logs',
                        dest='log_dir')
    parser.add_argument("--no-cv",
                        help="Turn off cross-validation and grid search.",
                        default=False,
                        action='store_true',
                        dest='no_crossvalidation')
    parser.add_argument("--cv-verbosity",
                        help=("The verbosity level of the Cross-Validation grid"
                              " search. The higher, the more verbose the grid"
                              " search is. 0 disables output."),
                        default=1,
                        type=int,
                        choices=[0, 1, 2],
                        dest='cv_verbosity')
    parser.add_argument("--model-params", "--p",
                        help=("Allows setting model parameters for the method"
                              " used. This should be a string with a python"
                              " expression containing a datastructure similar"
                              " to the grid_param argument to the cross "
                              "validation grid search, but the values doesn't have to be sequences. ",
                              "It will be used instead of the default grid_params."),
                        dest='model_params')
    parser.add_argument("--random-state",
                        help=("Give a start seed for random stuff. Ensures repeatability between runs. "
                              "If set to 'None', The random functions won't be seeded (using default initialization)"),
                        dest='random_state',
                        default='1729')
    args_dict = vars(parser.parse_args())

    ## Since we use 'None' to turn of constant seeding, we use eval here instead of just parsing the argument as an int
    args_dict['random_state'] = eval(args_dict['random_state'])
    return args_dict


def main():
    args_dict = get_cli_args()

    if args_dict['random_state'] is not None:
        np.random.seed(args_dict['random_state'])


    file_components = [args_dict['feature_type'],
                       args_dict['method'],
                       'frame_length_{}'.format(args_dict['frame_length'])]
    optional_file_components = {'standardized': args_dict['do_standardize'],
                                'sliding_frames': args_dict['sliding_frames'],
                                'segment_split': args_dict['do_segment_split'],
                                'downsampled': args_dict['do_downsample']}

    if args_dict['model_params'] is not None:
        args_dict['model_params'] = fix_model_params(args_dict['model_params'])

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    ## Setup loging stuff, this removes 'log_dir' from the dictionary
    setup_logging(timestamp, file_components, optional_file_components, args_dict)

    logging.info("Starting training with the following arguments: {}".format(args_dict))
    run_batch_classification(timestamp=timestamp,  **args_dict)


if __name__ == '__main__':
    main()
