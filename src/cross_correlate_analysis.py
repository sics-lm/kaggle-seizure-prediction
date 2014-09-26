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
import cross_correlate

from matplotlib.backends.backend_pdf import PdfPages

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
    """Returns the distributions of time deltas vs. cross correalation values"""
    delta_dist = defaultdict(list)
    for filename, correlations in file_correlations.items():
        for (channel_i, channel_j), frames in correlations.items():
            for (window_start, window_end), time_deltas in sorted(frames.items()):
                if ((segment_start is not None and window_end <= segment_start) or
                    (segment_end is not None and window_start >= segment_end)):
                    continue
                delta_dist[channel_i, channel_j].append(sorted(time_deltas))
    return delta_dist

def plot_delta_t_distributions(file_correlations, output, channels_per_plot=1, **kwargs):
    delta_dists = get_delta_t_distributions(file_correlations, **kwargs)
    figs = []
    pp = PdfPages(output)

    for i, (channel_pair, distributions) in enumerate(sorted(delta_dists.items())):
        fig = plt.figure(dpi=300)
        #Distributions is a list of lists: [[(delta_t, corrvalue)]] #We should plot the distributions as stacked plots,
        #We extract the times from the first distribution
        x = [delta_t for (delta_t, corrvalue) in distributions[0]]
        ys = [[corrvalue for delta_t, corrvalue in dist] for dist in distributions]
        ##This will probably be very hard to see, we might want to add the ys together instead
        plt.stackplot(x, ys)
        label = "{} {}".format(*channel_pair)
        print("Plotting: {}".format(label))
        #y = np.sum(ys, axis=0)
        #plt.plot(x,y,label=label)
        #plt.legend()
        plt.suptitle(label)
        plt.ylim((0,1))
        plt.xlabel("Time shift (s)")
        plt.ylabel("Correlation")
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
