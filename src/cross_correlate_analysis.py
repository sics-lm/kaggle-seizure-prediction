#!/usr/bin/env python
"""
Module for calculating the cross correlation between channels.
"""
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fileutils
import cross_correlate


def get_delta_ts(file_correlations, segment_start=None, segment_end=None):
    """Extract the delta_t:s from the correlations"""
    delta_ts = defaultdict(list)
    for filename, correlations in file_correlations.items():
        for (channel_i, channel_j), frames in correlations.items():
            for (window_start, window_end), time_deltas in sorted(frames.items()):
                if ((segment_start is not None and window_end <= segment_start) or
                    (segment_end is not None and window_start >= segment_end)):
                    continue
                (delta_t, correlation) = max(time_deltas, key=lambda x: x[1])
                delta_ts[channel_i, channel_j].append(delta_t)
    return delta_ts

def plot_delta_ts(file_correlations, output=None, channels_per_plot=4, **kwargs):
    delta_ts = get_delta_ts(file_correlations, **kwargs)
    figs = []
    if output is not None:
        pp = PdfPages(output)

    for i, (channel_pair, delta_t_list) in enumerate(sorted(delta_ts.items())):
        if i % channels_per_plot == 0:
            fig = plt.figure(dpi=300)

        plt.plot(delta_t_list, label="{} {}".format(*channel_pair))
        #If this is the last channel for this figure, or the very last channel, we do the final stuff
        next_i = i + 1
        if next_i % channels_per_plot == 0 or next_i == len(delta_ts):
                plt.legend()
                if output is not None:
                    pp.savefig(fig, dpi=300, papertype='a2')
                else:
                    figs.append(fig)
                plt.close(fig)
    if output is not None:
        pp.close()
    else:
        figs.append(fig)
    return figs


def get_delta_t_distributions(file_correlations, segment_start=None, segment_end=None):
    """Returns the distributions of time deltas vs. cross correalation values."""
    interictal_delta_dist = defaultdict(list)
    preictal_delta_dist = defaultdict(list)
    for filename, correlations in file_correlations.items():
        for (channel_i, channel_j), frames in correlations.items():
            for (window_start, window_end), time_deltas in sorted(frames.items()):
                if ((segment_start is not None and window_end <= segment_start) or
                    (segment_end is not None and window_start >= segment_end)):
                    continue
                if 'preictal' in filename.lower():
                    preictal_delta_dist[channel_i, channel_j].append(sorted(time_deltas))
                elif 'interictal' in filename.lower():
                    interictal_delta_dist[channel_i, channel_j].append(sorted(time_deltas))
    return interictal_delta_dist, preictal_delta_dist


def plot_delta_t_distributions(file_correlations, output, channels_per_plot=1, **kwargs):
    interictal_delta_dists, preictal_delta_dists = get_delta_t_distributions(file_correlations, **kwargs)
    figs = []
    pp = PdfPages(output)
    channels = interictal_delta_dists.keys()
    for i, channels in enumerate(sorted(channels)):
        fig = plt.figure(dpi=300)

        #Distributions is a list of lists: [[(delta_t, corrvalue)]] #We should plot the distributions as stacked plots,
        #We extract the times from the first distribution
        interictal_dist = interictal_delta_dists[channels]
        preictal_dist = preictal_delta_dists[channels]

        interictal_x = [delta_t for (delta_t, corrvalue) in interictal_dist[0]]
        interictal_ys = [[corrvalue for delta_t, corrvalue in dist] for dist in interictal_dist]

        preictal_x = [delta_t for (delta_t, corrvalue) in preictal_dist[0]]
        preictal_ys = [[corrvalue for delta_t, corrvalue in dist] for dist in preictal_dist]

        ##This will probably be very hard to see, we might want to add the ys together instead
        #plt.stackplot(x, ys)

        label = "{} {}".format(*channels)
        print("Plotting: {}".format(label))

        interictal_y = np.sum(interictal_ys, axis=0)
        interictal_norm = np.sum(interictal_y) # Calculate a constant to divide the interictal_y's with, so the integral is 1
        plt.plot(interictal_x,interictal_y/interictal_norm, label="Interictal")

        preictal_y = np.sum(preictal_ys, axis=0)
        preictal_norm = np.sum(preictal_y) # Calculate a constant to divide the preictal_y's with, so the integral is 1
        plt.plot(preictal_x,preictal_y/preictal_norm, label="Preictal")
        plt.legend()
        plt.suptitle(label)

        plt.xlabel("Time shift (s)")
        plt.ylabel("Normalized Correlation")
        pp.savefig(fig, dpi=300, papertype='a0')
        plt.close(fig)
    pp.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Script for doing analysis on the cross correlation from previously created csv files.")

    parser.add_argument("--output", help="The output to plot to.", metavar="FILENAME")
    parser.add_argument("--segment-start", help="If this argument is supplied, only windows after this time in seconds will be used.", type=float, metavar="SECONDS")
    parser.add_argument("--segment-end", help="If this argument is supplied, only windows before this time will be used.", type=float, metavar="SECONDS")
    parser.add_argument("--channels-per-plot", help="How many channels should each plot have", type=int)
    parser.add_argument("csv_files", help="The csv files to be plotted, chan be files, directories or combinations thereof", nargs="+")
    parser.add_argument("--mode", help="Which kind of plot to be produced", choices=["delta_times", "delta_dists"])
    #parser.add_argument("--channels", help="Selects a subset of the channels to use.")

    args = parser.parse_args()
    files = sorted(filter(lambda x: '.csv' in x, fileutils.expand_paths(args.csv_files)))

    correlations = dict()
    for f in files:
        try:
            correlations[f] = cross_correlate.read_csv(f)
        except:
            print("Error reading {} as a csv".format(f))
            raise
    if args.mode == 'delta_times':
        plot_delta_ts(correlations, output=args.output, segment_start=args.segment_start, segment_end=args.segment_end,
                  channels_per_plot=args.channels_per_plot)
    elif args.mode == 'delta_dists':
        plot_delta_t_distributions(correlations, output=args.output, segment_start=args.segment_start, segment_end=args.segment_end, channels_per_plot=args.channels_per_plot)
