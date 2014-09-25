#!/usr/bin/env python
"""
Module for producing spectra using the Welch method as provided by scipy.
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

def calculate_cross_correlations(s, delta_t, channels=None, window_length=None,
                                 segment_start=None, segment_end=None,
                                 workers=None, csv_writer=None):
    """Calculates the maximum cross-correlation of all pairs of channels in the segment s.
    *delta_t* is the time lag in seconds to do the cross correlations over, that is, the cross correlation will be
    over the time shifts (channel_0-delta_t * channel_1, ..., channel_0+delta_t * channel_1).
    The optional argument *channels* can be used to decide which channels should be included.
    If *window_length* is supplied, the data will be divided into windows of *window_length* seconds.
    *segment_start* and *segment_stop* can be given as times in second to only work on a part of the segment.
    """
    if channels is None:
        channels = s.get_channels()

    correlations = defaultdict(dict)

    #calculate what min_t and max_t are in samples
    sample_delta = int(delta_t * s.get_sampling_frequency())

    if segment_start is None:
        segment_start = 0
    if segment_end is None:
        segment_end = s.get_duration()

    for i, channel_i in enumerate(channels[:-1]):
        for channel_j in channels[i+1:]:
            if window_length is not None:
                for window_start in np.arange(segment_start, segment_end, window_length):
                    window_end = window_start + window_length
                    window_i = s.get_channel_data(channel_i, window_start, window_end)
                    window_j = s.get_channel_data(channel_j, window_start, window_end)
                    #maximum_crosscorr = maximum_crosscorelation(window_i, window_j, sample_delta)
                    #correlations[channel_i, channel_j][window_start, window_end] = maximum_crosscorr
                    if len(window_i) > 2:
                        #We skip strange boundry cases where the slice is too small to be useful
                        jobs.append((channel_i, channel_j, window_start, window_end, window_i, window_j, sample_delta))
            else:
                segment_i = s.get_channel_data(channel_i, segment_start, segment_end)
                segment_j = s.get_channel_data(channel_j, segment_start, segment_end)
                #maximum_crosscorr = maximum_crosscorelation(segment_i, segment_j, sample_delta)
                #correlations[channel_i, channel_j][segment_start, segment_end] = maximum_crosscorr
                if len(segment_i) > 2:
                    #We skip strange boundry cases where the slice is too small to be useful
                    jobs.append((channel_i, channel_j, segment_start, segment_end, segment_i, segment_j, sample_delta))
    if workers > 1:
        print("Workers are: ", workers)
        with multiprocessing.Pool(processes=workers) as pool:
            for result in pool.imap_unordered(worker_function, jobs):
                channel_i, channel_j, window_start, window_end, best_delta, correlation = result
                correlations[channel_i, channel_j][window_start, window_end] = (best_delta, correlation)
                if csv_writer is not None:
                    csv_writer.writerow(dict(channel_i=channel_i, channel_j=channel_j, start_sample=window_start,
                                             end_sample=window_end, t_offset=best_delta, correlation=correlation))
            pool.close()
    else:
        for result in map(worker_function, jobs):
            channel_i, channel_j, window_start, window_end, best_delta, correlation = result
            correlations[channel_i, channel_j][window_start, window_end] = (best_delta, correlation)
            if csv_writer is not None:
                csv_writer.writerow(dict(channel_i=channel_i, channel_j=channel_j, start_sample=window_start,
                                         end_sample=window_end, t_offset=best_delta, correlation=correlation))
    return correlations

def worker_function(data):
    channel_i, channel_j, window_start, window_end, window_i, window_j, sample_delta = data
    (best_delta_t, correlation) = maximum_crosscorelation(window_i, window_j, sample_delta)
    return channel_i, channel_j, window_start, window_end, best_delta_t, correlation


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


def maximum_crosscorelation(x, y, sample_delta):
    """Returns the maximal normalized cross-correlation for the two sequences x and y. *sample_delta* is the most *x* will be
    shifted 'to the left' and 'to the right' of *y*."""

    current_max = 0
    best_t = None

    #normalization of the values are done with sqrt(corr(x,x) dot corr(y,y))
    C_xx = np.dot(x,x)/x.size
    C_yy = np.dot(y,y)/y.size

    norm_const = np.sqrt(C_xx * C_yy)

    for t in range(1, sample_delta + 1):
        # For the negative values of t, we flip the arguments to corr, that is, y is shifted 'to the right' of x
        C_yx = corr(y, x, t)
        c = abs(C_yx / norm_const)
        if c > current_max:
            current_max = c
            best_t = -t

    for t in range(0, sample_delta + 1):
        C_xy = corr(x, y, t)
        c = abs(C_xy / norm_const)
        if c > current_max:
            current_max = c
            best_t = t

    return (best_t, current_max)



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


def write_csv(correlations):
    for f, corrs in correlations.items():
        name, ext = os.path.splitext(f)
        csv_name = "{}_cross_correlation.csv".format(name)
        with open(csv_name, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames, delimiter='\t')
            writer.writeheader()
            for (channel_i, channel_j), frames in corrs.items():
                for (start_sample, end_sample), (t_offset, corr_val) in sorted(frames.items()):
                    row = dict(channel_i=channel_i, channel_j=channel_j, start_sample=start_sample,
                               end_sample=end_sample, t_offset=t_offset, correlation=corr_val)
                    writer.writerow(row)


def read_csv(correlation_file):
    correlations = defaultdict(dict)
    with open(correlation_file) as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            channel_i = row['channel_i']
            channel_j = row['channel_j']
            window_start = float(row['start_sample'])
            window_end = float(row['end_sample'])
            t_offset = int(row['t_offset'])
            correlation = float(row['correlation'])
            correlations[(channel_i, channel_j)][(window_start, window_end)] = (t_offset, correlation)
    return correlations


def plot_correlations(correlations, output):
    for f, corrs in correlations.items():
        corrmap = []
        fig = plt.figure()

        for (channel_i, channel_j), frames in corrs.items():
            #Every channel pair becomes a row in the image
            corrdata = [float(correlation) for (window_start, window_end), (t_offset, correlation) in sorted(frames.items())]
            corrmap.append(corrdata)
        heatmap = plt.imshow(corrmap)
        fig.colorbar(heatmap)
    fig.savefig(output)


def get_csv_name(f, csv_directory):
    name, ext = os.path.splitext(f)
    if csv_directory is not None:
        basename = os.path.basename(name)
        name = os.path.join(csv_directory, basename)
    csv_name = "{}_cross_correlation.csv".format(name)
    return csv_name

if __name__ == '__main__':
    #test()
    #exit()
    #fileutils.process_segments(example_segments(), process_segment)
    #plot_welch_spectra(example_segments(), '../example.pdf')
    #exit(0)

    import argparse
    parser = argparse.ArgumentParser(description="Calculates the cross-correlation between the channels in the given segments.")

    parser.add_argument("--segments", help="The files to process. This can either be the path to a matlab file holding the segment or a directory holding such files.", nargs='+', metavar="SEGMENT_FILE")
    parser.add_argument("--plot", help="Plots the correlations as a heatmap to the supplied file.", metavar="FILENAME")
    parser.add_argument("--write-csv", help="Writes the cross-correlations to csv files. If no value is given for the output directory, the files will be collected to the same place as the input file(s).", nargs='?', metavar="CSV_OUTPUT_DIRECTORY")
    parser.add_argument("--read-csv", help="Reads the data from the given csv files instead of generating it from segments. Useful in combination with '--plot'.", nargs='+')
    parser.add_argument("--time-delta", help="Time delta in seconds to use for the cross-correlations. May be a floating point number.", type=float, default=0)
    parser.add_argument("--window-length", help="If this argument is supplied, the cross correlation will be done on windows of this length in seconds. If this argument is omitted, the whole segment will be used.", type=float)
    parser.add_argument("--segment-start", help="If this argument is supplied, only the segment after this time will be used.", type=float)
    parser.add_argument("--segment-end", help="If this argument is supplied, only the segment before this time will be used.", type=float)
    parser.add_argument("--workers", help="The number of worker processes used for calculating the cross-correlations.", type=int, default=1)
    #parser.add_argument("--channels", help="Selects a subset of the channels to use.")

    args = parser.parse_args()

    channels = None
    if args.segments:
        files = filter(lambda x: '.mat' in x, sorted(fileutils.expand_paths(args.segments)))
        if args.write_csv:
            correlations = dict()
            for f in files:
                csv_name = get_csv_name(f, args.write_csv)
                with open(csv_name, 'w') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames, delimiter='\t')
                    writer.writeheader()

                    correlations[f] = calculate_cross_correlations(segment.Segment(f),
                                                     args.time_delta, window_length=args.window_length,
                                                     channels=channels,
                                                     segment_start=args.segment_start,
                                                     segment_end=args.segment_end,
                                                     csv_writer=writer,
                                                     workers=args.workers)
        else:
            #Don't write to csv, only useful for directly plotting or testing
            correlations = { f: calculate_cross_correlations(segment.Segment(f),
                                                     args.time_delta, window_length=args.window_length,
                                                     channels=channels,
                                                     segment_start=args.segment_start,
                                                     segment_end=args.segment_end,
                                                     workers=args.workers) for f in files }


    elif args.read_csv:
        files = filter(lambda x: '.csv' in x, sorted(fileutils.expand_paths(args.read_csv)))
        correlations = { f: read_csv(f) for f in files}

    #if args.write_csv:
    #     write_csv(correlations)

    if args.plot:
        plot_correlations(correlations, args.plot)


