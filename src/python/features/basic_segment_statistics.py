#!/usr/bin/env python

"""Module for getting and plotting some basic statstics of segments"""
from __future__ import absolute_import
import glob
import os
import os.path
import random
import math
from collections import defaultdict

import pandas as pd
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

from ..datasets import segment

try:
    plt.style.use('ggplot')
except AttributeError:
    pass


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


def median_absolute_deviation(data, subject_median=None, axis=0):
    """A robust estimate of the standard deviation"""
    # The scaling factor for estimating the standard deviation from the MAD
    c = scipy.stats.norm.ppf(3/4)
    if isinstance(data, pd.DataFrame):
        if subject_median is None:
            subject_median = data.median(axis=axis)
        median_residuals = data - subject_median
        mad = median_residuals.abs().median(axis=axis)
    else:
        if subject_median is None:
            subject_median = np.median(data, axis=axis)
        median_residuals = data - subject_median[:, np.newaxis]  # deviation between median and data
        mad = np.median(np.abs(median_residuals), axis=axis)
    return mad/c


def load_and_transform_segments(feature_folder, glob_suffix='*', methods=None, processes=1):
    """Applies the each of the expressions in *methods* to each segment globbed by glob_pattern in feature_folder.
    Args:
        feature_folder: A folder containing the mat files to load.
        methods: A list of string values or dictionary with string values and string names. The string values should be
        expressions which can be 'evaled' when concatenated to a dataframe object. eg: ['.mean()'] will be concatenated
        and evaluateded as 'eval(\'df.mean()\')'
        If methods is a dictionary, the keys will be used to name the corresponding index of the dataframe, otherwise
        the same expression as is evaluated will be used.
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
                    expr = method.format(dataframe='seg.dataframe')
                    transformed[name][basename] = eval(expr)
            else:
                for method in methods:
                    expr = method.format(dataframe='seg.dataframe')
                    transformed[method][basename] = eval(expr)

        for method_name, results in transformed.items():
            segment_names, segment_series = zip(*results.items())
            index = pd.MultiIndex.from_product([[segment_class], list(segment_names)])
            method_frame = pd.DataFrame(list(segment_series), index=index)
            class_results[method_name].append( method_frame )

    complete_frame = pd.concat([pd.concat(frames) for method, frames in class_results.items()],
                               keys=class_results.keys(), names=['metric', 'class', 'segment'])
    return complete_frame


def get_default_methods(subset=None):
    methods = {'max': '{dataframe}.max()',
               'min': '{dataframe}.min()',
               'median': '{dataframe}.median()',
               'mad': 'median_absolute_deviation({dataframe})',
               'mean': '{dataframe}.mean()',
               'mean absolute deviation': '{dataframe}.mad()',
               'std': '{dataframe}.std()',
               'sem': '{dataframe}.sem()',
               'skew': '{dataframe}.skew()',
               'kurtosis': '{dataframe}.kurtosis()',
               'absolute median': '{dataframe}.abs().median()',
               'absolute mean': '{dataframe}.abs().mean()',
               'absolute max': '{dataframe}.abs().max()'}
    if subset is not None:
        methods = {method: expr for method, expr in methods.items() if method in subset}
    return methods


def process_subject(feature_folder, methods = None,
                    subset=None,
                    glob_suffix='*', processes=1):
    """Calculates a bunch of statistics for the different classes of the subject in feature_folder"""
    if methods is None:
        methods = get_default_methods(subset=subset)

    class_results = load_and_transform_segments(feature_folder, methods=methods, glob_suffix=glob_suffix, processes=processes)
    return class_results


def calculate_statistics(feature_folder, csv_directory, processes=1, glob_suffix='*', subset=None):
    """Calculates statistics for all the segment files in *feature_folder* and saves it to a file in csv_directory"""
    slashed_path = os.path.join(feature_folder, '') ## This makes sure the path has a trailing '/', so that os.dirname will give the whole path
    _, subject_folder_name = os.path.split(os.path.dirname(slashed_path))
    filename = "{}_segments_statistics.csv".format(subject_folder_name)

    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)
    csv_path = os.path.join(csv_directory, filename)
    class_results = process_subject(feature_folder, glob_suffix=glob_suffix, processes=processes, subset=subset)
    class_results.to_csv(csv_path, sep='\t', float_format='%11.8f')



def read_stats(stat_file, metrics=None, use_cache=True):
    if metrics is not None:
        # We convert the metrics to a sorted tuple, so that it's hashable
        metrics = tuple(sorted(metrics))
    cache = read_stats.cache
    if use_cache and stat_file in cache and metrics in cache[stat_file]:
        return cache[stat_file][metrics]
    else:
        stats_df = read_stats_csv(stat_file)
        if metrics is not None:
            stats_df = stats_df.loc[metrics]
        reshaped = stats_df.reset_index(['metric', 'class', 'segment']).drop('class', axis=1).pivot('segment', 'metric')
        reshaped.sortlevel(axis=1)
        read_stats.cache[stat_file][metrics] = reshaped
        return reshaped
read_stats.cache = defaultdict(dict)


def read_folder(stats_folder, metrics=None):
    if metrics is None:
        metrics = ['absolute mean', 'absolute median', 'kurtosis', 'skew', 'std']
    glob_pattern = os.path.join(stats_folder, '*segments_statistics.csv')
    files = glob.glob(glob_pattern)
    if files:
        stats = pd.concat([read_stats(stat_file, metrics) for stat_file in files])
        return stats
    else:
        raise IOError("No segment statistics file in folder {}".format(stats_folder))


def get_subject_metric(stats_df, metric_name, aggregator='{dataframe}.median()', channel_ordering=None, use_cache=True):
    """
    Returns the metric given by stats df as a NDArray-like of shape (n_channels, 1)
    :param stats_df: The statistics dataframe aquired from read_stats.
    :param metric_name: The metric we wan't to select.
    :param aggregator: A string with an expression used to aggregate the per segment statistic to a single statistic for
    the whole subject.
    :param channel_ordering: An optional ordered sequence of channel names, which will ensure that the outputted
    statistics vector has the same order as the segment which the statistic should be applied on.
    :return: A NDArray of shape (n_channels, 1) where each element along axis 0 correspond to a statistic for that channel.
    """
    cache = get_subject_metric.cache
    assert isinstance(stats_df, pd.DataFrame)
    if use_cache and id(stats_df) in cache and (metric_name, aggregator) in cache[id(stats_df)]:
        return cache[id(stats_df)][(metric_name, aggregator)]

    # The stats dataframes have a 2-level column index, where the first level are the channel names and the seconde
    # the metric name. To get the metric but keep the channels we slice the first level with all the entries using
    # slice(None), this is equivalent to [:] for regular slices.
    if channel_ordering is None:
        segment_metrics = stats_df.loc[:, (slice(None), metric_name)]
    else:
        segment_metrics = stats_df.loc[:, (channel_ordering, metric_name)]
    aggregated_metric = eval(aggregator.format(dataframe='segment_metrics'))
    added_axis = aggregated_metric[:,np.newaxis]
    cache[id(stats_df)][(metric_name, aggregator)] = added_axis
    return added_axis
get_subject_metric.cache = defaultdict(dict)


def plot_stats(stats_df, title=None, metrics=None):
    if metrics is None:
        metrics = ['absolute mean',
                   'absolute median',
                   'kurtosis',
                   'mad',
                   'max',
                   'mean',
                   'mean absolute deviation',
                   'median',
                   'min',
                   'sem',
                   'skew',
                   'std']
    rows = int(math.floor(math.sqrt(len(metrics))))
    cols = int(math.ceil(len(metrics)/rows))

    f, a = plt.subplots(rows, cols)
    for metric, ax in zip(metrics, a.flatten()):
        stats_df.xs(metric).plot(kind='bar', colormap='gist_rainbow', ax=ax)
        ax.get_legend().set_visible(False)
        ax.set_title(metric)
    if title is not None:
        f.suptitle(title)


def boxplot_metric(stats_df, metric):
     stats_df.xs(metric).reset_index('class').boxplot(by='class')


def read_stats_csv(stat_path):
    stats_df = pd.read_csv(stat_path, sep='\t', index_col=['metric', 'class', 'segment'])
    stats_df.sortlevel('metric', inplace=True)
    return stats_df

def read_subject_stats(stat_path):
    stats_df = read_stats_csv(stat_path)
    return stats_df.groupby(level=('metric', 'class')).mean()


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
    parser.add_argument("--metrics",
                        nargs='+',
                        help="A selection of statistics to collect",
                        choices=get_default_methods().keys(),
                        dest='subset')
    args = parser.parse_args()

    calculate_statistics(args.feature_folder, args.csv_directory, args.processes, args.glob_suffix, subset=args.subset)
