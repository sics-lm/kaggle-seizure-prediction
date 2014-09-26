#!/usr/bin/env python
"""
Module for calculating the cross correlation between channels.
"""
import numpy as np
import math
import scipy.signal
from collections import defaultdict
import matplotlib.pyplot as plt
import multiprocessing
import os.path
import csv

import fileutils
import segment

csv_fieldnames = ['channel_i', 'channel_j', 'start_sample', 'end_sample', 't_offset', 'correlation']

def calculate_cross_correlations(s, time_delta_config, channels=None, window_length=None,
                                 segment_start=None, segment_end=None,
                                 workers=None, csv_writer=None, all_time_deltas=False):
    """Calculates the maximum cross-correlation of all pairs of channels in the segment s.
    *time_delta_config* is a time delta specification dictionary with channel pairs as keys. The special pair
    ('default', 'default') will be used for any channel pair not present in the dictionary.
    The optional argument *channels* can be used to decide which channels should be included.
    If *window_length* is supplied, the data will be divided into windows of *window_length* seconds.
    *segment_start* and *segment_stop* can be given as times in second to only work on a part of the segment.
    *workers* is the number of processes to use for calculating the cross-correlations. *csv_writer* is the
    object which writes the data to file. If *all_time_deltas* is True, all correlations for all time steps will be
    included in the files, otherwise only the maximum correlation between two channels will be included.

    """
    if channels is None:
        channels = s.get_channels()

    frequency = s.get_sampling_frequency()

    if segment_start is None:
        segment_start = 0
    if segment_end is None:
        segment_end = s.get_duration()

    jobs = []
    for i, channel_i in enumerate(channels[:-1]):
        for channel_j in channels[i+1:]:
            if (channel_i, channel_j) in time_delta_config:
               time_delta_begin, time_delta_end, time_delta_step = time_delta_config[channel_i, channel_j]
            elif (channel_j, channel_i) in time_delta_config:
                time_delta_begin, time_delta_end, time_delta_step = time_delta_config[channel_j, channel_i]
            else:
                time_delta_begin, time_delta_end, time_delta_step = time_delta_config['default', 'default']

            #convert the time shifts range from seconds to discrete sample steps, the step range must be at least 1
            time_delta_range = (int(time_delta_begin*frequency),
                                int(time_delta_end*frequency),
                                max(int(time_delta_step*frequency), 1))

            if window_length is not None:
                for window_start in np.arange(segment_start, segment_end, window_length):
                    window_end = window_start + window_length
                    window_i = s.get_channel_data(channel_i, window_start, window_end)
                    window_j = s.get_channel_data(channel_j, window_start, window_end)
                    if len(window_i) > 2:
                        #We skip strange boundry cases where the slice is too small to be useful
                        jobs.append((channel_i, channel_j, window_start, window_end, window_i, window_j, time_delta_range, all_time_deltas))
            else:
                segment_i = s.get_channel_data(channel_i, segment_start, segment_end)
                segment_j = s.get_channel_data(channel_j, segment_start, segment_end)
                if len(segment_i) > 2:
                    #We skip strange boundry cases where the slice is too small to be useful
                    jobs.append((channel_i, channel_j, segment_start, segment_end, segment_i, segment_j, time_delta_range, all_time_deltas))
    if workers > 1:
        with multiprocessing.Pool(processes=workers) as pool:
            for result in pool.imap_unordered(worker_function, jobs):
                channel_i, channel_j, window_start, window_end, time_deltas = result
                #time_deltas is a list of (delta_t, correlation) values, if all_time_deltass is False, it will be the maximum correlation
                for delta_t, correlation in time_deltas:
                    t_offset = delta_t / frequency
                    csv_writer.writerow(dict(channel_i=channel_i, channel_j=channel_j, start_sample=window_start,
                                             end_sample=window_end, t_offset=t_offset, correlation=correlation))
            pool.close()
    else:
        for result in map(worker_function, jobs):
            channel_i, channel_j, window_start, window_end, time_deltas = result
            #time_deltas is a list of (delta_t, correlation) values, if all_time_deltass is False, it will be the maximum correlation
            for delta_t, correlation in time_deltas:
                t_offset = delta_t / frequency
                csv_writer.writerow(dict(channel_i=channel_i, channel_j=channel_j, start_sample=window_start,
                                         end_sample=window_end, t_offset=t_offset, correlation=correlation))


def worker_function(data):
    channel_i, channel_j, window_start, window_end, window_i, window_j, sample_delta, all_time_deltas = data
    time_deltas = maximum_crosscorelation(window_i, window_j, sample_delta, all_time_deltas=all_time_deltas)
    return channel_i, channel_j, window_start, window_end, time_deltas

def corr(x,y, t):
    """
    Calculate the correlation between the equal length arrays x and y at time lag t. t should be greater or equal
    to zero. The formula used is:
    C(x,y)(t) = 1/(N-t) * sum_{i = 0}^{N-t}(x[i+t]y[i])
    """
    # We slice y to only include the elements which will overlap with x.
    # if x = [1,2,3,4,5] and y = [6,7,8,9,10], with
    # a t=3 we want them to line up so that x[3] is multiplied with y[0]:
    # x = [1,2,3,4,5]
    # y =       [6, 7, 8, 9, 10]
    # We do this by slicing x so [4,5] are left and y so that [6,7] are left and then multiply the two arrays
    N = x.size
    if t > 0:
        x_sliced = x[t:]
        y_sliced = y[:N-t]
        sig_corr = np.dot(x_sliced, y_sliced)
    elif t == 0:
        sig_corr = np.dot(x, y)
    else:
        raise ValueError("The time shift has to be greater or equal to t")
    return sig_corr.take(0)/(N -t)


def maximum_crosscorelation(x, y, time_delta_range, all_time_deltas=False):
    """Returns the maximal normalized cross-correlation for the two sequences x and y. *sample_delta* is the most *x* will be
    shifted 'to the left' and 'to the right' of *y*. If *all_time_deltas* is True, all time deltas and their response will be included."""

    current_max = 0
    best_t = None

    #normalization of the values are done with sqrt(corr(x,x) dot corr(y,y))
    C_xx = np.dot(x,x)/x.size
    C_yy = np.dot(y,y)/y.size

    norm_const = np.sqrt(C_xx * C_yy)

    time_deltas = []
    time_delta_begin, time_delta_end, time_delta_step = time_delta_range
    for t in range(time_delta_begin, 0, time_delta_step):
        # For the negative values of t, we flip the arguments to corr, that is, y is shifted 'to the right' of x
        C_yx = corr(y, x, -t)
        c = abs(C_yx / norm_const)
        if all_time_deltas:
                time_deltas.append((t, c))
        if c > current_max:
            current_max = c
            best_t = -t


    for t in range(0, time_delta_end + 1, time_delta_step):
        C_xy = corr(x, y, t)
        c = abs(C_xy / norm_const)
        if all_time_deltas:
                time_deltas.append((t, c))

        if c > current_max:
            current_max = c
            best_t = t

    if all_time_deltas:
        return time_deltas
    else:
        return [(best_t, current_max)]



def example_segments():
    segments = ['../data/Dog_1/Dog_1_preictal_segment_0001.mat', '../data/Dog_1/Dog_1_interictal_segment_0001.mat']
    return segments


def test():
    x = np.sin((np.arange(0,100) * np.pi))
    y = np.sin((np.arange(0,100) * np.pi) + 4)
    print(maximum_crosscorelation(x, y, 5))
    # s = segment.Segment(fileutils.get_preictal_files('../data/Dog_1')[0])
    # channels = s.get_channels()[:2]
    # corrs = calculate_window_cross_correlation(s, 1, 100, 110, channels)
    # print(corrs)


def write_csv(correlations, window_length, time_delta):
    for f, corrs in correlations.items():
        name, ext = os.path.splitext(f)
        csv_name = "{}_cross_correlation_{}s_{}dt.csv".format(name, window_length, time_delta)
        with open(csv_name, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames, delimiter='\t')
            writer.writeheader()
            for (channel_i, channel_j), frames in corrs.items():
                for (start_sample, end_sample), (t_offset, corr_val) in sorted(frames.items()):
                    row = dict(channel_i=channel_i, channel_j=channel_j, start_sample=start_sample,
                               end_sample=end_sample, t_offset=t_offset, correlation=corr_val)
                    writer.writerow(row)


def read_csv(correlation_file):
    correlations = defaultdict(lambda: defaultdict(list))
    with open(correlation_file) as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            channel_i = row['channel_i']
            channel_j = row['channel_j']
            window_start = float(row['start_sample'])
            window_end = float(row['end_sample'])
            t_offset = float(row['t_offset'])
            correlation = float(row['correlation'])
            correlations[(channel_i, channel_j)][(window_start, window_end)].append((t_offset, correlation))
    return correlations


def get_csv_name(f, csv_directory, window_length=None, time_delta_config=None):
    name, ext = os.path.splitext(f)
    if csv_directory is not None:
        basename = os.path.basename(name)
        name = os.path.join(csv_directory, basename)
    csv_name = "{}_cross_correlation".format(name)
    if window_length is not None:
        csv_name += "_{}s".format(window_length)

    return csv_name + '.csv'


def setup_time_delta(time_delta_begin, time_delta_end, time_delta_step, time_delta_config_file):
    """Returns a timedelta specification"""
    time_delta_config = dict()
    time_delta_config['default', 'default'] = (min(time_delta_begin, -time_delta_begin), time_delta_end, time_delta_step)
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


if __name__ == '__main__':
    #test()
    #exit()
    #fileutils.process_segments(example_segments(), process_segment)
    #plot_welch_spectra(example_segments(), '../example.pdf')
    #exit(0)

    import argparse
    parser = argparse.ArgumentParser(description="Calculates the cross-correlation between the channels in the given segments. Saves the results to a csv file per segment file.")

    parser.add_argument("segments", help="The files to process. This can either be the path to a matlab file holding the segment or a directory holding such files.", nargs='+', metavar="SEGMENT_FILE")
    parser.add_argument("--csv-directory", help="Directory to write the csv files to, if omitted, the files will be written to the same directory as the segment")
    parser.add_argument("--time-delta-begin", help="Time delta in seconds to shift 'left' for the cross-correlations. May be a floating point number. Should be a negative number, if not it will be negated.", type=float, default=0)
    parser.add_argument("--time-delta-end", help="Time delta in seconds to shift 'right' for the cross-correlations. May be a floating point number.", type=float, default=0)
    parser.add_argument("--time-delta-step", help="Time delta range step in seconds.", type=float, default=0)
    parser.add_argument("--time-delta-config", help="A file holding time delta values for the different channels.")
    parser.add_argument("--all-time-deltas", help="Includes the time delta vs. correlation for all time deltas, and not just the maimal, that is, all the correlations for all time steps in the time delta range. Warning: this might use a lot of memory. A factor of (time_delta_begin - time_delta_end)/time_step more memory.", action='store_true')
    parser.add_argument("--window-length", help="If this argument is supplied, the cross correlation will be done on windows of this length in seconds. If this argument is omitted, the whole segment will be used.", type=float)
    parser.add_argument("--segment-start", help="If this argument is supplied, only the segment after this time will be used.", type=float)
    parser.add_argument("--segment-end", help="If this argument is supplied, only the segment before this time will be used.", type=float)
    parser.add_argument("--workers", help="The number of worker processes used for calculating the cross-correlations.", type=int, default=1)

    #parser.add_argument("--channels", help="Selects a subset of the channels to use.")

    args = parser.parse_args()

    time_delta_config = setup_time_delta(args.time_delta_begin, args.time_delta_end, args.time_delta_step, args.time_delta_config)

    channels = None
    files = filter(lambda x: '.mat' in x, sorted(fileutils.expand_paths(args.segments)))
    correlations = dict()
    for f in files:
        csv_name = get_csv_name(f, args.csv_directory, args.window_length, time_delta_config)
        with open(csv_name, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames, delimiter='\t')
            writer.writeheader()

            correlations[f] = calculate_cross_correlations(segment.Segment(f),
                                             time_delta_config, window_length=args.window_length,
                                             channels=channels,
                                             segment_start=args.segment_start,
                                             segment_end=args.segment_end,
                                             csv_writer=writer,
                                             workers=args.workers,
                                             all_time_deltas=args.all_time_deltas)
