#!/usr/bin/env python

"""Module for getting and plotting some basic statstics of segments"""
import glob
import os
import os.path
import multiprocessing
import random
from collections import defaultdict

import pandas as pd

import segment

def segments_means(segments):
    """Returns a data frame with the means of the given segments"""
    return segments.dataframe.groupby(level='filename').mean()


def segments_std(segments):
    """Returns a data frame with the standard deviation of the segments signal"""
    return segments.dataframe.groupby(level='filename').std()


def segments_abs_means(segments):
    """Returns a data frame with the means of the absolute values of the given segment channels"""
    return segments.dataframe.groupby(level='filename').aggregate(lambda series: series.abs().mean())


def get_filenames(feature_folder, glob_pattern, sample_size=None):
    files = glob.glob(os.path.join(feature_folder,glob_pattern))
    if sample_size is not None and sample_size < len(files):
        files = random.sample(files, sample_size)
    return files


def load_segments(feature_folder, glob_pattern='*.mat', sample_size=None):
    files = get_filenames(feature_folder, glob_pattern, sample_size)
    segments = segment.DFSegment.from_mat_files(files)
    return segments


def load_and_transform_segments(feature_folder, glob_suffix='*', methods=None):
    """Applies the each of the expressions in *methods* to each segment globbed by glob_pattern in feature_folder.
    Args:
        feature_folder: A folder containing the mat files to load.
        methods: A list of string values or dictionary with string values and string names. The string values should be expressions which can be 'evaled' when concatenated to a dataframe object. eg: ['.mean()'] will be concatenated and evaluateded as 'eval(\'df.mean()\')'
        If methods is a dictionary, the keys will be used to name the corresponding index of the dataframe, otherwise the same expression as is evaluated will be used.
    """
    class_results = defaultdict(list)

    for segment_class in ['preictal', 'interictal', 'test']:
        glob_pattern = "*{}*{}*".format(segment_class, glob_suffix)
        files = get_filenames(feature_folder, glob_pattern)
        transformed = defaultdict(dict)

        for f in files:
            print("Processing {}".format(f))
            basename = os.path.basename(f)
            seg = segment.DFSegment.from_mat_file(f)
            if isinstance(methods, dict):
                for name, method in methods.items():
                    expr = 'seg.dataframe{method}'.format(method=method)
                    transformed[name][basename] = eval(expr)
            else:
                for method in methods:
                    expr = 'seg.dataframe{method}'.format(method=method)
                    transformed[method][basename] = eval(expr)

        for method_name, results in transformed.items():
            segment_names, segment_series = zip(*results.items())
            index = pd.MultiIndex.from_product([[segment_class], list(segment_names)])
            method_frame = pd.DataFrame(list(segment_series), index=index)
            class_results[method_name].append( method_frame )

    complete_frame = pd.concat([pd.concat(frames) for method, frames in class_results.items()],
                               keys=class_results.keys(), names=['metric', 'class', 'segment'])
    return complete_frame


def process_subject(feature_folder, methods = {'median': '.median()',
                                               'mean': '.mean()',
                                               'mad' : '.mad()',
                                               'std': '.std()',
                                               'sem': '.sem()',
                                               'skew': '.skew()',
                                               'kurtosis': '.kurtosis()',
                                               'absolute median': '.abs().median()',
                                               'absolute mean': '.abs().mean()'},
                    glob_suffix='*'):
    """Calculates a bunch of statistics for the different classes of the subject in feature_folder"""
    class_results = load_and_transform_segments(feature_folder, methods=methods, glob_suffix=glob_suffix)
    return class_results


def calculate_statistics(feature_folder, csv_directory, processes=1, glob_suffix='*'):
    """Calculates statistics for all the segment files in *feature_folder* and saves it to a file in csv_directory"""
    slashed_path = os.path.join(feature_folder, '') ## This makes sure the path has a trailing '/', so that os.dirname will give the whole path
    _, subject_folder_name = os.path.split(os.path.dirname(slashed_path))
    filename = "{}_segments_statistics.csv".format(subject_folder_name)

    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)
    csv_path = os.path.join(csv_directory, filename)
    class_results = process_subject(feature_folder, glob_suffix=glob_suffix)
    class_results.to_csv(csv_path, sep='\t', float_format='%11.4f')


def read_stats(stat_file, metrics=None):
    stats_df = pd.read_csv(stat_file, sep='\t', index_col=['metric', 'class', 'segment'])
    assert isinstance(stats_df, pd.DataFrame)
    stats_df.sortlevel('metric', inplace=True)
    if metrics is not None:
        stats_df = stats_df.loc[metrics]
    reshaped = stats_df.reset_index(['metric', 'class', 'segment']).drop('class', axis=1).pivot('segment', 'metric')
    return reshaped


def read_folder(stats_folder, metrics=['absolute mean', 'absolute median', 'kurtosis', 'skew', 'std']):
    glob_pattern = os.path.join(stats_folder, '*segments_statistics.csv')
    files = glob.glob(glob_pattern)
    if files:
        stats = pd.concat([read_stats(stat_file, metrics) for stat_file in files])
        return stats
    else:
        raise FileNotFoundError("No segment statistics file in folder {}".format(stats_folder))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="""Script for generating a bunch of statistics about the segments""")

    parser.add_argument("feature_folder",
                        help="""The folder containing the features to be analyzed""")
    parser.add_argument("--glob-suffix",
                        help="""Unix-style glob patterns to select which files to run the analyziz over. This suffix will  be appendend t othe class label (eg 'interictal') so should not be expressed here. Be use to encase the pattern in " so it won't be expanded by the shell.""", dest='glob_suffix', default='*')
    parser.add_argument("--processes",
                        help="How many processes should be used for parellelized work.",
                        dest='processes',
                        default=1,
                        type=int)
    parser.add_argument("--csv-directory",
                        help="Which directory the statistics CSV file be written to.",
                        dest='csv_directory',
                        default='../../data/segment_statistics')

    args = parser.parse_args()

    calculate_statistics(args.feature_folder, args.csv_directory, args.processes, args.glob_suffix)
