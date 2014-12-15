#!/usr/bin/env python
"""
Module for calculating the cross correlation between channels.
"""
from __future__ import division
from __future__ import absolute_import

from collections import defaultdict
import os.path
import csv
import re

import numpy as np

from . import feature_extractor
from ..datasets import fileutils

csv_fieldnames = ['channel_i', 'channel_j', 'start_sample', 'end_sample', 't_offset', 'correlation']

channel_pattern = re.compile(r'(?:[a-zA-Z0-9]*_)*(c[0-9]*|[A-Z]*_[0-9]*)$')


def convert_channel_name(name):
    """Pass"""
    match = re.match(channel_pattern, name)
    if match:
        return match.group(1) or match.group(2)
    else:
        return name


def calculate_cross_correlations(s, time_delta_config, channels=None, window_length=None,
                                 segment_start=None, segment_end=None, all_time_deltas=False,
                                 old_csv_format=False):
    """
    Calculates the maximum cross-correlation of all pairs of channels in the segment *s*.

    :param s: The segment object to calculate the correlations from.
    :param time_delta_config: A dictionary with channel pairs as keys, and time delta range triplets as values.
                              The triplets define the time lags to use for the given channel pairs and should be given
                              in seconds. For example (-0.2, 0.2, 0.1) will calculate the correlation at the time lags
                              [-0.2, -0.1, 0.0, 0.1, 0.2]. The dictionary can have a channel pair ('default', 'default)
                              which will be used for any pair which isn't explicitly specified.
    :param channels: A collection of channels to calculate the correlation over. Can be used to constrain the
                     calculation to a subset of the channels.
    :param window_length: The window length in seconds to calculate the correlation over.
    :param segment_start: The time in seconds in the segment to start the calculations from. Used to constrain the
                          calculations to a part of the whole segment.
    :param segment_end: The time in seconds in the segment to end the calculations. Used to constrain the
                        calculations to a part of the whole segment.
    :param all_time_deltas: If True, all time lags correlation will be kept. If False, only the maximal correlation is
                            kept.
    :param old_csv_format: If True, the old inefficient CSV file format is used. If False, the new much more space
                           efficient format is used. Both formats encode the same information.
    :return: A list of dictionaries, where each dictionary correspond to a row of correlation data at a specific time
             lag. The contents of the dictionaries depend on the csv format used. The old format has every dictionary
             as a single channel pair at a single window and time lag, the new format has the dictionaries as all the
             channel pairs for a specific window and time lag.
    """
    if channels is None:
        channels = s.get_channels()

    frequency = s.get_sampling_frequency()

    if segment_start is None:
        segment_start = 0
    if segment_end is None:
        segment_end = s.get_duration()

    results = defaultdict(list)

    for i, channel_i in enumerate(channels[:-1]):
        for channel_j in channels[i + 1:]:
            if (channel_i, channel_j) in time_delta_config:
                time_delta_begin, time_delta_end, time_delta_step = time_delta_config[channel_i, channel_j]
            elif (channel_j, channel_i) in time_delta_config:
                time_delta_begin, time_delta_end, time_delta_step = time_delta_config[channel_j, channel_i]
            else:
                time_delta_begin, time_delta_end, time_delta_step = time_delta_config['default', 'default']

            # Convert the time shifts range from seconds to discrete sample steps, the step range must be at least 1
            time_delta_range = (int(time_delta_begin * frequency),
                                int(time_delta_end * frequency),
                                max(int(time_delta_step * frequency), 1))

            if window_length is not None:
                windows = [(window_start, window_start + window_length)
                           for window_start
                           in np.arange(segment_start, segment_end, window_length)]
            else:
                windows = [(segment_start, segment_end)]

            for window_start, window_end in windows:
                window_i = s.get_channel_data(channel_i, window_start, window_end)
                window_j = s.get_channel_data(channel_j, window_start, window_end)

                # We skip strange boundary cases where the slice is too small to be useful
                if len(window_i) > 2:
                    time_deltas = maximum_crosscorrelation(window_i, window_j, time_delta_range, all_time_deltas)
                    # Time_deltas is a list of (delta_t, correlation) values, if all_time_deltas is False,
                    # it will be the maximum correlation
                    for delta_t, correlation in time_deltas:
                        t_offset = delta_t / float(frequency)

                        results[(channel_i, channel_j)].append((window_start,
                                                                window_end,
                                                                t_offset,
                                                                correlation))

    # We should return a list of dictionaries, where the keys of each dictionary are the same and will be the columns
    # of the csv file
    if old_csv_format:
        table = []
        for (channel_i, channel_j), result_tuples in sorted(results.items()):
            for start_sample, end_sample, t_offset, correlation in sorted(result_tuples):
                table.append(dict(channel_i=channel_i,
                                  channel_j=channel_j,
                                  start_sample=start_sample,
                                  end_sample=end_sample,
                                  t_offset=t_offset,
                                  correlation=correlation))
        return table
    else:
        start_sample_grouped = defaultdict(lambda: defaultdict(dict))

        # We group the correlations by start_sample and t_offset
        for (channel_i, channel_j), result_tuples in results.items():
            channel_i_name = convert_channel_name(channel_i)
            channel_j_name = convert_channel_name(channel_j)
            pair_name = channel_i_name + ':' + channel_j_name
            for start_sample, end_sample, t_offset, correlation in result_tuples:
                start_sample_grouped[(start_sample, end_sample)][t_offset].update({pair_name: correlation})

        table = []
        for (start_sample, end_sample), t_offsets in sorted(start_sample_grouped.items()):
            for t_offset, channel_correlations in sorted(t_offsets.items()):
                row = dict(start_sample=start_sample,
                           end_sample=end_sample,
                           t_offset=t_offset)
                row.update(channel_correlations)
                table.append(row)
        return table


def maximum_crosscorrelation(x, y, time_delta_range, all_time_deltas=False):
    """
    Returns the normalized cross-correlation for the two sequences x and y at time lags specified by *time_delta_range*.

    :param x: The first vector to use in the cross correlation.
    :param y: The second vector to use in the cross correlation.
    :param time_delta_range: A triple specifying the time lag range. The format is the same as for the range built-in
                             function but the upper bound is included.
                             For example (-20, 20, 5) will calculate the time lags at [-20, -15, -10, -5, 0, 5, 10, 15].
    :param all_time_deltas: If True, all correlation values will be kept and returned during calculations. If False
                            only the time lag with the maximal correlation is kept.
    :return: A list of (time_lag, correlation) pairs. If all_time_deltas is False, only the maximal correlation pair is
             kept.
    """

    current_max = -1
    best_t = None

    # Normalization of the values are done with sqrt(corr(x,x) dot corr(y,y))
    c_xx = np.dot(x, x) / x.size
    c_yy = np.dot(y, y) / y.size

    norm_const = np.sqrt(c_xx * c_yy)

    time_deltas = []
    time_delta_begin, time_delta_end, time_delta_step = time_delta_range
    for t in range(time_delta_begin, 0, time_delta_step):
        # For the negative values of t, we flip the arguments to corr, that is, y is shifted 'to the right' of x
        c_yx = corr(y, x, -t)
        c = abs(c_yx / norm_const)
        if all_time_deltas:
            time_deltas.append((t, c))
        if c > current_max:
            current_max = c
            best_t = -t

    for t in range(0, time_delta_end + 1, time_delta_step):
        c_xy = corr(x, y, t)
        c = abs(c_xy / norm_const)
        if all_time_deltas:
            time_deltas.append((t, c))

        if c > current_max:
            current_max = c
            best_t = t

    if all_time_deltas:
        return time_deltas
    else:
        return [(best_t, current_max)]


def corr(x, y, t):
    """
    Calculate the correlation between the equal length arrays x and y at time lag t. t should be greater or equal
    to zero. The formula used is:
    C(x,y)(t) = 1/(n-t) * sum_{i = 0}^{n-t}(x[i+t]y[i])
    """
    # We slice y to only include the elements which will overlap with x.
    # if x = [1,2,3,4,5] and y = [6,7,8,9,10], with
    # a t=3 we want them to line up so that x[3] is multiplied with y[0]:
    # x = [1,2,3,4,5]
    # y =       [6, 7, 8, 9, 10]
    # We do this by slicing x so [4,5] are left and y so that [6,7] are left and then multiply the two arrays
    n = x.size
    if t > 0:
        x_sliced = x[t:]
        y_sliced = y[:n - t]
        sig_corr = np.dot(x_sliced, y_sliced)
    elif t == 0:
        sig_corr = np.dot(x, y)
    else:
        raise ValueError("The time shift has to be greater or equal to t")
    return sig_corr.take(0) / (n - t)


def get_csv_name(f, csv_directory, window_length=None):
    name, ext = os.path.splitext(f)
    if csv_directory is not None:
        basename = os.path.basename(name)
        name = os.path.join(csv_directory, basename)
    csv_name = "{}_cross_correlation".format(name)
    if window_length is not None:
        csv_name += "_{}s".format(window_length)

    return csv_name + '.csv'


def csv_naming_function(segment_path, output_dir, window_length=None, **kwargs):
    """Wrapper for get_csv_name for use as a feature_extrator naming function."""
    if fileutils.get_subject(output_dir) is None:
        subject = fileutils.get_subject(segment_path)
        output_dir = os.path.join(output_dir, subject)

    return get_csv_name(segment_path, output_dir, window_length)


def setup_time_delta(time_delta_begin, time_delta_end, time_delta_step, time_delta_config_file):
    """Returns a timedelta specification"""
    time_delta_config = dict()
    time_delta_config['default', 'default'] = (
        min(time_delta_begin, -time_delta_begin), time_delta_end, time_delta_step)
    if time_delta_config_file is not None:
        with open(time_delta_config_file) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter='\t')
            for row in csv_reader:
                channel_i = row['channel_i']
                channel_j = row['channel_j']
                begin = float(row['begin'])
                end = float(row['end'])
                step = float(row['step'])
                time_delta_config[channel_i, channel_j] = (min(begin, -begin), end, step)
    return time_delta_config


def extract_features(segment_paths,
                     output_dir,
                     workers=1,
                     resample_frequency=None,
                     normalize_signal=False,
                     window_size=5,
                     only_missing_files=True,
                     # Arguments for calculate_cross_correlations
                     time_delta_config=None,
                     time_delta_begin=0,
                     time_delta_end=0,
                     time_delta_step=0,
                     channels=None,
                     segment_start=None,
                     segment_end=None,
                     all_time_deltas=False,
                     old_csv_format=False):
    time_delta_config = setup_time_delta(time_delta_begin, time_delta_end, time_delta_step, time_delta_config)
    feature_extractor.extract(feature_folder=segment_paths,
                              extractor_function=calculate_cross_correlations,
                              # Arguments for feature_extractor.extract
                              output_dir=output_dir,
                              workers=workers,
                              naming_function=csv_naming_function,
                              normalize_signal=normalize_signal,
                              only_missing_files=only_missing_files,
                              resample_frequency=resample_frequency,
                              # Arguments for calculate_cross_correlations
                              time_delta_config=time_delta_config,
                              window_length=window_size,
                              channels=channels,
                              segment_start=segment_start,
                              segment_end=segment_end,
                              all_time_deltas=all_time_deltas,
                              old_csv_format=old_csv_format, )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=("Calculates the cross-correlation between the channels in the given segments. "
                     "Saves the results to a csv file per segment file."))

    parser.add_argument("segments",
                        help=("The files to process. This can either be the path to a matlab file holding the "
                              "segment or a directory holding such files."),
                        nargs='+', metavar="SEGMENT_FILE")
    parser.add_argument("--csv-directory",
                        help=("Directory to write the csv files to, if omitted, the files will be written to the same "
                              "directory as the segment"))
    parser.add_argument("--time-delta-begin",
                        help=("Time delta in seconds to shift 'left' for the cross-correlations. May be a floating "
                              "point number. Should be a negative number, if not it will be negated."),
                        type=float, default=0)
    parser.add_argument("--time-delta-end",
                        help=("Time delta in seconds to shift 'right' for the cross-correlations. May be a floating "
                              "point number."),
                        type=float, default=0)
    parser.add_argument("--time-delta-step", help="Time delta range step in seconds.", type=float, default=0)
    parser.add_argument("--time-delta-config", help="A file holding time delta values for the different channels.")
    parser.add_argument("--all-time-deltas",
                        help=("Includes the time delta vs. correlation for all time deltas, and not just the maimal, "
                              "that is, all the correlations for all time steps in the time delta range. Warning: this "
                              "might use a lot of memory. A factor of (time_delta_begin - time_delta_end)/time_step "
                              "more memory."),
                        action='store_true')
    parser.add_argument("--window-length",
                        help=("If this argument is supplied, the cross correlation will be done on windows of this "
                              "length in seconds. If this argument is omitted, the whole segment will be used."),
                        type=float)
    parser.add_argument("--segment-start",
                        help="If this argument is supplied, only the segment after this time will be used.", type=float)
    parser.add_argument("--segment-end",
                        help="If this argument is supplied, only the segment before this time will be used.",
                        type=float)
    parser.add_argument("--workers", help="The number of worker processes used for calculating the cross-correlations.",
                        type=int, default=1)
    parser.add_argument("--old-csv-format", help="Use the old CSV format where the channel pairs are rows",
                        action='store_true',
                        dest='old_csv_format')
    parser.add_argument("--new-segment-format",
                        help="Use the old Segment format where the data is accesses through a numpy array",
                        action='store_false',
                        dest='new_segment_format',
                        default=False)
    parser.add_argument("--only-missing-files",
                        help="Should the feature extractor only extract features for non-missing files",
                        default=False,
                        action='store_true',
                        dest='only_missing_files')
    parser.add_argument("--resample-frequency", help="The frequency to resample to,",
                        type=float,
                        dest='resample_frequency')
    parser.add_argument("--normalize-signal",
                        help="Setting this flag will normalize the channels based on the subject median and MAD",
                        default=False,
                        action='store_true',
                        dest='normalize_signal')

    args = parser.parse_args()

    channels = None

    extract_features(segment_paths=args.segments,
                     output_dir=args.csv_directory,
                     workers=args.workers,
                     resample_frequency=args.resample_frequency,
                     normalize_signal=args.normalize_signal,
                     window_size=args.window_length,
                     only_missing_files=args.only_missing_files,
                     # Arguments for calculate_cross_correlations
                     time_delta_config=args.time_delta_config,
                     time_delta_begin=args.time_delta_begin,
                     time_delta_end=args.time_delta_end,
                     time_delta_step=args.time_delta_step,
                     segment_start=args.segment_start,
                     channels=channels,
                     segment_end=args.segment_end,
                     all_time_deltas=args.all_time_deltas,
                     old_csv_format=args.old_csv_format)


if __name__ == '__main__':
    main()

