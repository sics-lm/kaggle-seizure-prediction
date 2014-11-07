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
import features_combined
import dataset
import seizure_modeling
import fileutils
import submissions


def run_batch_classification(feature_folders,
                             timestamp,
                             submission_file=None,
                             frame_length=1,
                             sliding_frames=False,
                             rebuild_data=False,
                             feature_type='cross-correlation',
                             processes=1,
                             **kwargs):
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

    all_scores = []
    for feature_dict in load_features(feature_folders,
                                      feature_type=feature_type,
                                      frame_length=frame_length,
                                      sliding_frames=sliding_frames,
                                      rebuild_data=rebuild_data,
                                      processes=processes):
        kwargs.update(feature_dict)  # Adds the content of feature dict to the keywords for run_classification
        segment_scores = run_classification(processes=processes, **kwargs)
        score_dict = segment_scores.to_dict()['preictal']
        all_scores.append(score_dict)

    if submission_file is None:
        name_components = ["submission"]
        name_components.append(feature_type)
        name_components.append(kwargs['method'])
        if kwargs['do_standardize']:
            name_components.append("standardized")
        name_components.append(str(timestamp))
        filename = '_'.join(name_components) + '.csv'
        submission_file = os.path.join('..', '..', 'submissions', filename)

    logging.info("Saving submission scores to {}".format(submission_file))
    with open(submission_file, 'w') as fp:
        submissions.write_scores(all_scores, output=fp)


def load_features(feature_folders,
                  feature_type='cross-correlations',
                  frame_length=1,
                  sliding_frames=False,
                  rebuild_data=False,
                  processes=1):
    """
    Loads the features from the list of paths *feature_folder*. Returns an
    iterator of dictionaries, where each dictionary has the keys 'subject_folder',
    'interictal_data', ''preictal_data' and 'unlabeled_data'.

    Args:
        feature_folders: A list of paths to folders containing features.
                         The features in these folders will be combined into
                         three data frames.
        feature_type: A string describing the type of features to use. If
                      'wavelets' is supplied, the feature files will be loaded
                      as wavelets. If 'cross-correlations' or 'xcorr' is
                      supplied, the features will be loaded as cross-correlation
                      features. If 'combined' is supplied, the path of the
                      feature folders will be used to determine which features
                      it contains, and the results will be combined column-wise
                      into longer feature vectors.
        frame_length: The desired frame length in windows of the features.
        sliding_frames: If True, the training feature frames will be generated
                        by a sliding window, greatly increasing the number of
                        generated frames.
        processes: The number of processes to use for parallel processing.
    Returns:
        A generator object which gives a dictionary with features for every call
        to next. The dictionary contains the keys 'subject_folder',
        'interictal_data', 'preictal_data' and 'unlabeled_data'.
    """
    feature_folders = fileutils.expand_folders(feature_folders)

    if feature_type == 'wavelets' or feature_type == 'cross-correlations' or feature_type == 'xcorr':
        if feature_type == 'wavelets':
            feature_module = wavelet_classification
        else:
            feature_module = correlation_convertion
        for feature_folder in feature_folders:
            interictal, preictal, unlabeled = feature_module.load_data_frames(feature_folder,
                                                                              rebuild_data=rebuild_data,
                                                                              processes=processes,
                                                                              frame_length=frame_length,
                                                                              sliding_frames=sliding_frames)
            yield dict(interictal_data=interictal,
                       preictal_data=preictal,
                       unlabeled_data=unlabeled,
                       subject_folder=feature_folder)

    elif feature_type == 'combined':
        combined_folders = fileutils.group_folders(feature_folders)
        for subject, combo_folders in combined_folders.items():
            ## We create an output folder which is based on the subject name
            subject_folder = os.path.join('..', '..', 'data', 'combined', subject)
            if not os.path.exists(subject_folder):
                os.makedirs(subject_folder)

            interictal, preictal, unlabeled = dataset.load_data_frames(combo_folders,
                                                                       load_function=features_combined.load,
                                                                       find_features_function=fileutils.find_grouped_feature_files,
                                                                       rebuild_data=rebuild_data,
                                                                       processes=processes,
                                                                       frame_length=frame_length,
                                                                       sliding_frames=sliding_frames,
                                                                       output_folder=subject_folder)
            yield dict(interictal_data=interictal,
                       preictal_data=preictal,
                       unlabeled_data=unlabeled,
                       subject_folder=subject_folder)
    else:
        raise NotImplementedError("No feature loading method implemented for feature type {}".format(feature_type))


def run_classification(interictal_data,
                       preictal_data,
                       unlabeled_data,
                       subject_folder,
                       training_ratio=.8,
                       model_file=None,
                       rebuild_model=False,
                       do_downsample=False,
                       downsample_ratio=2.0,
                       do_standardize=False,
                       method="logistic",
                       do_segment_split=False,
                       processes=4,
                       csv_directory=None,
                       do_refit=True,
                       cv_verbosity=2):
    logging.info("Running classification on folder {}".format(subject_folder))
    if do_standardize:
        logging.info("Standardizing variables.")
        interictal_data, preictal_data, unlabeled_data = dataset.scale(interictal_data,
                                                                       preictal_data,
                                                                       unlabeled_data,
                                                                       inplace=True)
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
                                             do_downsample=do_downsample,
                                             downsample_ratio=downsample_ratio,
                                             do_segment_split=do_segment_split,
                                             training_ratio=training_ratio,
                                             processes=processes,
                                             do_standardize=do_standardize,
                                             cv_verbosity=cv_verbosity)
        if model_file is None:
            #Create a new filename based on the model method and the
            #date
            model_basename = "model_{}_{}.pickle".format(method, timestamp)
            model_file = os.path.join(subject_folder, model_basename)
        with open(model_file, 'wb') as fp:
            pickle.dump(model, fp)

    if do_refit:
        logging.info("Refitting model with held-out data.")
        model = seizure_modeling.refit_model(interictal_data,
                                             preictal_data,
                                             model,
                                             do_downsample=do_downsample,
                                             downsample_ratio=downsample_ratio,
                                             do_segment_split=do_segment_split)

    if csv_directory is None:
        csv_directory = subject_folder
    scores = write_scores(csv_directory, unlabeled_data, model, timestamp=timestamp)
    logging.info("Finnished with classification on folder {}".format(subject_folder))

    return scores


def write_scores(csv_directory, test_data, model, timestamp=None):
    """
    Writes the model prediction scores for the segments of *test_data* to a csv
    file.
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
                                 'knn'],
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
    parser.add_argument("--cv-verbosity",
                        help=("The verbosity level of the Cross-Validation grid"
                              " search. The higher, the more verbose the grid"
                              " search is. 0 disables output."),
                        default=1,
                        type=int,
                        choices=[0, 1, 2],
                        dest='cv_verbosity')

    args_dict = vars(parser.parse_args())
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    ## Setup loging stuff, this removes 'log_dir' from the dictionary
    setup_logging(timestamp, args_dict)

    logging.info("Starting training with the following arguments: {}".format(args_dict))

    run_batch_classification(timestamp=timestamp, **args_dict)
