"""
Module for running the feature extraction and model training.
"""
from __future__ import absolute_import
from __future__ import print_function

import json
import os.path
import datetime

from python.features import hills_features, wavelets, cross_correlate
from python.classification import classification_pipeline


def extract_features(settings):
    """
    Extract features based on the dictionary *settings*. The type of features extracted depends on the key
    'FEATURE_TYPE' and should be either 'xcorr', 'wavelets' or 'hills'.

    :param settings: A dictionary with settings. Usually created from the json file 'SETTINGS.json' in the project root
                     directory.
    :return: None. The features will be saved as csv files to the directory given by the key 'FEATURE_PATH' in the
             settings dictionary.
    """
    output_dir = settings['FEATURE_PATH']
    workers = settings['WORKERS']
    window_size = settings['FEATURE_SETTINGS']['WINDOW_LENGTH']
    frame_length = settings['FEATURE_SETTINGS']['FEATURE_WINDOWS']
    segment_paths = settings['TRAIN_DATA_PATH']
    if settings['FEATURE_TYPE'] == 'hills':
        hills_features.extract_features(segment_paths=segment_paths,
                                        output_dir=output_dir,
                                        workers=workers,
                                        window_size=settings['FEATURE_SETTINGS']['WINDOW_LENGTH'],
                                        feature_length_seconds=window_size*frame_length)

    elif settings['FEATURE_TYPE'] == 'xcorr':
        cross_correlate.extract_features(segment_paths=segment_paths,
                                         output_dir=output_dir,
                                         workers=workers,
                                         window_size=settings['FEATURE_SETTINGS']['WINDOW_LENGTH'])

    elif settings['FEATURE_TYPE'] == 'wavelets':
        wavelets.extract_features(segment_paths=segment_paths,
                                  output_dir=output_dir,
                                  workers=workers,
                                  window_size=settings['FEATURE_SETTINGS']['WINDOW_LENGTH'],
                                  feature_length_seconds=window_size*frame_length)


def train_model(settings):
    """
    Trains a SVM classifier using the features selected by the  *settings* dictionary.
    When fitted, the model will automatically be used to assign scores and a submission file will be generated in the
    folder given by 'SUBMISSION_PATH' in the *settings* dictionary.

    :param settings: A dictionary with settings. Usually created from the json file 'SETTINGS.json' in the project root
                     directory.
    :return: None. The model will be pickled to a file in the corresponding subject feature folder. A submission file
             will be written to the folder given by 'SUBMISSION_PATH' in the settings dictionary.
    """
    # classification_pipeline.train_models(feature_folders=[settings['FEATURE_PATH']],
    #                                      feature_type=settings['FEATURE_TYPE'],
    #                                      model_dir=settings['MODEL_PATH'],
    #                                      processes=settings['WORKERS'])
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    classification_pipeline.run_batch_classification(feature_folders=[settings['FEATURE_PATH']],
                                                     timestamp=timestamp,
                                                     submission_file=settings['SUBMISSION_PATH'],
                                                     frame_length=12,
                                                     feature_type=settings['FEATURE_TYPE'],
                                                     processes=settings['WORKERS'],
                                                     do_standardize=True,
                                                     no_crossvalidation=True,
                                                     rebuild_model=True,
                                                     method='svm',
                                                     model_params={'C': 500, 'gamma': 0})


def fix_settings(settings, root_dir):
    """
    Goes through the settings dictionary and makes sure the paths are correct.
    :param settings: A dictionary with settings, usually obtained from SETTINGS.json in the root directory.
    :param root_dir: The root path to which any path should be relative.
    :return: A settings dictionary where all the paths are fixed to be relative to the supplied root directory.
    """
    fixed_settings = dict()
    for key, setting in settings.items():
        if 'path' in key.lower():
            if isinstance(setting, str):
                setting = os.path.join(root_dir, setting)
            elif isinstance(setting, list):
                setting = [os.path.join(root_dir, path) for path in setting]
        fixed_settings[key] = setting
    return fixed_settings


def get_settings(settings_path):
    """
    Reads the given json settings file and makes sure the path in it are correct.
    :param settings_path: The path to the json file holding the settings.
    :return: A dictionary with settings.
    """
    with open(settings_path) as settings_fp:
        settings = json.load(settings_fp)
    root_dir = os.path.dirname(settings_path)
    fixed_settings = fix_settings(settings, root_dir)
    return fixed_settings


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extracts features and trains model")
    parser.add_argument("settings", help="Path to the SETTINGS.json to use for the training")

    args = parser.parse_args()
    settings = get_settings(args.settings)
    print("Extracting Features")
    extract_features(settings)
    print("Training model")
    train_model(settings)


if __name__ == '__main__':
    main()
