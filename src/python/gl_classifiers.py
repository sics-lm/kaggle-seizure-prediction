from __future__ import print_function

import graphlab as gl # WARNING: Graphlab supports python2 only
import graphlab.toolkits.model_parameter_search as mps
import glob
import os.path
from classification_pipeline import get_latest_model
from time import strftime, localtime
import random
import sys


# def run_batch_classification(feature_folder_root="../../data/wavelets",
#                              rebuild_data=False, training_ratio=1,
#                              rebuild_model=False, do_downsample=False,
#                              method="glm", do_segment_split=False):

#     for subject in ("Dog_1", "Dog_2", "Dog_3",
#                     "Dog_4", "Dog_5", "Patient_1",
#                     "Patient_2"):
#         run_classification(
#             os.path.join(feature_folder_root, subject),
#             rebuild_data=rebuild_data,
#             training_ratio=training_ratio, rebuild_model=rebuild_model,
#             do_down_sample=do_downsample, method=method,
#             do_segment_split=do_segment_split)

def load_feature_files(feature_folder,
                       class_name,
                       rebuild_data,
                       file_pattern="extract_features_for_segment.csv"):
    cache_file = os.path.join(feature_folder, class_name)
    if rebuild_data or not os.path.exists(cache_file):
        full_pattern = "*{}*{}".format(class_name, file_pattern)
        glob_pattern = os.path.join(feature_folder, full_pattern)

        complete_sframe = gl.SFrame.read_csv(
            glob_pattern, header=False, column_type_hints=float)

        complete_sframe.save(cache_file)
    else:
        complete_sframe = gl.load_sframe(cache_file)
    return complete_sframe


def load_sframes(feature_folder, rebuild_data=False,
                 file_pattern="extract_features_for_segment.csv"):
    preictal = load_feature_files(
        feature_folder, class_name="preictal", file_pattern=file_pattern,
        rebuild_data=rebuild_data)
    interictal = load_feature_files(
        feature_folder, class_name="interictal", file_pattern=file_pattern,
        rebuild_data=rebuild_data)
    test = load_feature_files(
        feature_folder, class_name="test", file_pattern=file_pattern,
        rebuild_data=rebuild_data)

    preictal['preictal'] = 1
    interictal['preictal'] = 0

    complete = interictal.append(preictal)

    return complete, test

def load_single_sframe(filename):
    return gl.SFrame.read_csv(filename, header=False, column_type_hints=float)

def split_experiment_data(complete, training_ratio=0.8, do_downsample=True, downsample_ratio=2.0, seed=None):
    # Figure how many positive and negative samples we have in the complete
    # training set.
    train_preictal = complete[complete['preictal'] == 1]
    train_interictal = complete[complete['preictal'] == 0]

    interictal_samples = train_interictal.shape[0]
    preictal_samples = train_preictal.shape[0]

    # Get approximatelly downsample_ratio * preictal_samples from the
    # interictal_samples sframe

    desired_interictal = downsample_ratio * preictal_samples
    interictal_ratio = desired_interictal / float(interictal_samples)

    if seed == None:
        seed = int(random.randrange(0, int(sys.maxint/10e10)))

    downsampled_interictal = train_interictal.sample(
        interictal_ratio, seed=seed)

    downsampled_complete = downsampled_interictal.append(train_preictal)
    # OK to re-use seed
    print("Original interictal samples: %d" % interictal_samples)
    print("Original preictal samples: %d" % preictal_samples)
    print("Desired interictal samples: %d" % desired_interictal)
    print("Interictal samples after downsampling: %d" % downsampled_interictal.shape[0])
    return downsampled_complete.random_split(training_ratio, seed=seed)

def train_model(data, method):
    target = 'preictal'

    if method == 'svm':
        return gl.svm_classifier.create(data, target)
    elif method == 'boosted-trees':
        return gl.boosted_trees_classifier.create(data, target)
    elif method == 'logistic-regression':
        return gl.logistic_classifier.create(data, target)
    elif method == 'neural-networks':
        return gl.neuralnet_classifier.create(data, target)

def evaluate_model(model, test_data):
    return model.evaluate(test_data)

def model_search(env, train_path, test_path, method, model_file):
    target = 'preictal'

    if method == 'svm':
        mps(env, svm_classifier.create, train_path, model_file, test_path)
    elif method == 'boosted-trees':
        mps(env, boosted_trees_classifier.create, train_path, model_file, test_path)
    elif method == 'logistic-regression':
        mps(env, logistic_classifier.create, train_path, model_file, test_path)
    elif method == 'neural-networks':
        mps(env, neuralnet_classifier.create, train_path, model_file, test_path)


def run_classification(feature_folder, rebuild_data=False, training_ratio=.8,
                       rebuild_model=False, model_file=None,
                       do_downsample=False, method="svm"):

    print("Running classification on folder {}".format(feature_folder))

    complete, test = load_sframes(feature_folder, rebuild_data=rebuild_data)

    training_data, test_data = split_experiment_data(
        complete, training_ratio=training_ratio, do_downsample=do_downsample)

    if model_file is None or not rebuild_model:
        model_file = get_latest_model(feature_folder, "model*.gl")
        if model_file is None:
            rebuild_model = True

    # Complete test, parser options
    timestamp = strftime("%m-%d-%Y-%H.%M.%S", localtime())
    if rebuild_model:
        model = train_model(training_data, method=method)
        if model_file is None:
            #Create a new filename based on the model method and the
            #date
            model_basename = "model_{}_{}.gl".format(method, timestamp)
            model_file = os.path.join(feature_folder, model_basename)
        model.save(model_file)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="""Script for running the classification pipeline""")
    parser.add_argument("feature_folder_root", help="""The folder containing the features collected in subject subfolders""", default="../../data/cross_correlation")
    parser.add_argument("--rebuild-data", action='store_true', help="Should the dataframes be re-read from the csv feature files", dest='rebuild_data')
    parser.add_argument("--training-ratio", type=float, default=0.8, help="What ratio of the data should be used for training", dest='training_ratio')
    parser.add_argument("--rebuild-model", action='store_true', help="Should the model be rebuild, or should a cached version (if available) be used.", dest='rebuild_model')
    parser.add_argument("--do-downsample", action='store_true', help="should class imbalance be solved by downsampling the majority class", dest='do_downsample')
    parser.add_argument("--method", help="What model to use for learning", dest='method')

    args = parser.parse_args()
    run_batch_classification(**vars(args))