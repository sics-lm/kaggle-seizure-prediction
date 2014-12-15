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


def get_filenames(feature_folder, glob_pattern, sample_size=None):
    """
    Finds the all the files in the given feature folder which matches the glob pattern.
    :param feature_folder: The folder to search for files.
    :param glob_pattern: The glob pattern to use for finding files.
    :param sample_size: If given, restrict the number of files loaded to a sample of this size.
    :return: A list of files matching the glob pattern in the feature folder.
    """
    files = glob.glob(os.path.join(feature_folder, glob_pattern))
    if sample_size is not None and sample_size < len(files):
        files = random.sample(files, sample_size)
    return files


def load_segments(feature_folder, glob_pattern='*.mat', sample_size=None):
    """
    Loads segments from the given folder.

    :param feature_folder: The folder to search in.
    :param glob_pattern: The glob pattern to match files by.
    :param sample_size: If given, a sample of all possible files will be taken.
    :return: A DataFrame where all the given segments have been concatenated.
    """
    files = get_filenames(feature_folder, glob_pattern, sample_size)
    segments = segment.DFSegment.from_mat_files(files)
    return segments


def median_absolute_deviation(data, subject_median=None, axis=0):
    """
    Returns the median absolute deviation (MAD) for the given data.
    :param data: A DataFrame holding the data to calculate the MAD from.
    :param subject_median: If given, these will be used as the medians which the deviation is calculated from. If None
                           The median of the given data is used.
    :param axis: The axis to calculate over.
    :return: A Series object with the median absolute deviations over the given *axis* for the *data*.
    """
    """A robust estimate of the standard deviation"""
    # The scaling factor for estimating the standard deviation from the MAD
    c = scipy.stats.norm.ppf(3 / 4)
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
    return mad / c


def load_and_transform_segments(feature_folder, glob_suffix='*', metrics=None):
    """
    Applies each of the expressions in *methods* to each segment globbed by glob_pattern in feature_folder.

    :param feature_folder: The folder containing the mat signal files to load.
    :param glob_suffix:
    :param metrics: A list of string values or dictionary with string keys and values. The elements of the list or the
                    values of the dictionary should be expressions which are evaluated to produce a dataframe results
                    for a segment. The string can contain the formatting identifier '{dataframe}' which will be replaced
                    by the segment dataframe in the evaluation. An example of an expression is '{dataframe}.mean()'
                    which will return the mean of the dataframe.
                    If metrics is a dictionary, the keys should be strings which identifies the metric. For example:
                    metrics={'median absolute deviation': 'median_absolute_deviation({dataframe})'} would use
                    'median absolute deviation' to identify the results of the evaluated expression
                    'median_absolute_deviation({dataframe})'.
                    If methods is a list, the expression will be used as an identifier.
    :return: A DataFrame with the segment statistics. The DataFrame has a multi-level index, where the top level is
             given by the metric name (for example 'median absolute deviation' above). The seconds level is the
             class of the segment ('interictal', 'preictal' or 'test') and the final level is the segment name. The
             columns of the DataFrame correspond to the metric for the channels of the signal.
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
            if isinstance(metrics, dict):
                for name, metric in metrics.items():
                    expr = metric.format(dataframe='seg.dataframe')
                    transformed[name][basename] = eval(expr)
            else:
                for metric in metrics:
                    expr = metric.format(dataframe='seg.dataframe')
                    transformed[metric][basename] = eval(expr)

        for metric_name, results in transformed.items():
            segment_names, segment_series = zip(*results.items())
            index = pd.MultiIndex.from_product([[segment_class], list(segment_names)])
            method_frame = pd.DataFrame(list(segment_series), index=index)
            class_results[metric_name].append(method_frame)

    complete_frame = pd.concat([pd.concat(frames) for method, frames in class_results.items()],
                               keys=class_results.keys(), names=['metric', 'class', 'segment'])
    return complete_frame


def get_default_metrics(subset=None):
    """
    Returns a metrics dictionary with the default metrics.
    :param subset: If given as a list of strings, only those metrics whose key matches elements in this list will be
                   returned.
    :return: A dictionary of identifier to metric expressions as used by load_and_transform_segments.
    """
    metrics = {'max': '{dataframe}.max()',
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
        metrics = {metric: expr for metric, expr in metrics.items() if metric in subset}
    return metrics


def process_subject(feature_folder,
                    metrics=None,
                    subset=None,
                    glob_suffix='*'):
    """
    Calculates summary statistics for the segments in the given feature folder.
    :param feature_folder: The folder holding the matlab segments.
    :param metrics: An optional dictionary of identifier to expression mappings which are used to calculate the
                    statistics.
    :param subset: A subset of the default metrics to use.
    :param glob_suffix: The suffix to use for matching segment files.
    :return: A DataFrame with the segment statistics. The DataFrame has a multi-level index, where the top level is
             given by the metric name (for example 'median absolute deviation' above). The seconds level is the
             class of the segment ('interictal', 'preictal' or 'test') and the final level is the segment name. The
             columns of the DataFrame correspond to the metric for the channels of the signal.
    """

    if metrics is None:
        metrics = get_default_metrics(subset=subset)

    class_results = load_and_transform_segments(feature_folder, metrics=metrics, glob_suffix=glob_suffix)
    return class_results


def calculate_statistics(feature_folder, csv_directory, glob_suffix='*', subset=None):
    """
    Calculates statistics for segments in the given feature folder and saves them to a CSV file in the given csv
    directory.
    :param feature_folder: The folder to load segment files from.
    :param csv_directory: The directory to save the statistics csv file to.
    :param glob_suffix: A suffix to use for globbing files.
    :param subset: A subset of the default metrics to use.
    :return: None. The statistics are saved to a CSV file in *csv_directory* with the same name as the subject name
             derived from feature_folder.
    """
    # This makes sure the path has a trailing '/', so that os.dirname will give the whole path
    slashed_path = os.path.join(feature_folder, '')
    _, subject_folder_name = os.path.split(os.path.dirname(slashed_path))
    filename = "{}_segments_statistics.csv".format(subject_folder_name)

    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)
    csv_path = os.path.join(csv_directory, filename)
    class_results = process_subject(feature_folder, glob_suffix=glob_suffix, subset=subset)
    class_results.to_csv(csv_path, sep='\t', float_format='%11.8f')


def read_stats(stat_file, metrics=None, use_cache=True):
    """
    Reads a previously created statistics CSV file into a DataFrame.
    :param stat_file: The statistics file to load.
    :param metrics: The metrics to read.
    :param use_cache: If True, the metrics will be cached by the function, which speeds up the reading of the same
                      statistics file serially.
    :return: A DataFrame with the statistics from stat_file.
    """
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
    """
    Reads all segment statistics files from the given folder and returns the statistics as a single DataFrame.
    :param stats_folder: The folder to reat statistics files from.
    :param metrics: A subset of the metrics to load.
    :return: A DataFrame with all the statistics available in stats_folder.
    """
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
    Returns the metric given by stats df as a NDArray-like of shape (n_channels, 1), obtained by aggregating the
    given metric from the dataframe.
    :param stats_df: The statistics dataframe acquired from read_stats.
    :param metric_name: The metric to collect.
    :param aggregator: A string with an expression used to aggregate the per segment statistic to a single statistic for
                       the whole subject.
    :param channel_ordering: An optional ordered sequence of channel names, which will ensure that the outputted
    statistics vector has the same order as the segment which the statistic should be applied on.
    :param use_cache: If True, the metrics will be cached by the function so that calling it multiple times for the
                      same subject is fast.
    :return: A NDArray of shape (n_channels, 1) where each element along axis 0 correspond to the aggregated statistic
             that channel.
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
    added_axis = aggregated_metric[:, np.newaxis]
    cache[id(stats_df)][(metric_name, aggregator)] = added_axis
    return added_axis
get_subject_metric.cache = defaultdict(dict)


def plot_stats(stats_df, title=None, metrics=None):
    """
    Plot the statistics into nice plots.
    :param stats_df: The DataFrame with the statistics to plot.
    :param title: The title of the plot.
    :param metrics: A list of the metrics to include in the plot.
    :return: The matplotlib Figure object for this plot.
    """
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
    cols = int(math.ceil(len(metrics) / rows))

    f, a = plt.subplots(rows, cols)
    for metric, ax in zip(metrics, a.flatten()):
        stats_df.xs(metric).plot(kind='bar', colormap='gist_rainbow', ax=ax)
        ax.get_legend().set_visible(False)
        ax.set_title(metric)
    if title is not None:
        f.suptitle(title)
    return f


def boxplot_metric(stats_df, metric):
    """
    Boxplot the given metric by class.
    :param stats_df: The statistics DataFrame to use for plotting.
    :param metric: The metric to plot.
    :return: None. The plot will automatically 'show', since this is a call to the DataFrame.boxplot function.
    """
    stats_df.xs(metric).reset_index('class').boxplot(by='class')


def read_stats_csv(stat_path):
    """
    Reads a statistics CSV file as a data frame.
    :param stat_path: The file to read.
    :return: A DataFrame with the given metric.
    """
    stats_df = pd.read_csv(stat_path, sep='\t', index_col=['metric', 'class', 'segment'])
    stats_df.sortlevel('metric', inplace=True)
    return stats_df


def read_subject_stats(stat_path):
    """
    Returns the mean of all statistics grouped by class for the given statistics file.
    :param stat_path: The file to read statistics from.
    :return: A DataFrame with the mean of all the metrics, grouped by class.
    """
    stats_df = read_stats_csv(stat_path)
    return stats_df.groupby(level=('metric', 'class')).mean()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="""Script for generating a bunch of statistics about the segments""")

    parser.add_argument("feature_folder",
                        help="""The folder containing the features to be analyzed""")
    parser.add_argument("--glob-suffix",
                        help=("Unix-style glob patterns to select which files to run the analysis over."
                              " This suffix will  be appended to the class label (eg 'interictal') so should not be "
                              "expressed here. "
                              "Be sure to encase the pattern in \" so it won't be expanded by the shell."),
                        dest='glob_suffix', default='*')
    # parser.add_argument("--processes",
    #                     help="How many processes should be used for parellelized work.",
    #                     dest='processes',
    #                     default=1,
    #                     type=int)
    parser.add_argument("--csv-directory",
                        help="Which directory the statistics CSV file be written to.",
                        dest='csv_directory',
                        default='../../data/segment_statistics')
    parser.add_argument("--metrics",
                        nargs='+',
                        help="A selection of statistics to collect",
                        choices=get_default_metrics().keys(),
                        dest='subset')
    args = parser.parse_args()

    calculate_statistics(args.feature_folder, args.csv_directory, args.glob_suffix, args.subset)


if __name__ == '__main__':
    main()