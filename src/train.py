"""
Module for running the feature extraction and model training.
"""
from __future__ import absolute_import
from __future__ import print_function

import json
import os.path
import sys
import datetime

from python.features import hills_features, wavelets, cross_correlate
from python.classification import classification_pipeline


def extract_features(settings):
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
    # classification_pipeline.train_models(feature_folders=[settings['FEATURE_PATH']],
    #                                      feature_type=settings['FEATURE_TYPE'],
    #                                      model_dir=settings['MODEL_PATH'],
    #                                      processes=settings['WORKERS'])
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    classification_pipeline.run_batch_classification(feature_folders=[settings['FEATURE_PATH']],
                                                     timestamp=timestamp,
                                                     frame_length=12,
                                                     feature_type=settings['FEATURE_TYPE'],
                                                     processes=settings['WORKERS'],
                                                     do_standardize=True,
                                                     no_crossvalidation=True,
                                                     rebuild_model=True,
                                                     method='svm',
                                                     model_params={'C': 500, 'gamma': 0})


def fix_settings(settings, root_dir):
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
