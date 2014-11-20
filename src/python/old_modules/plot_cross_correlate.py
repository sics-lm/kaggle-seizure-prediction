#!/usr/bin/env python
"""
Module for plotting cross correlations.
"""
from collections import defaultdict

import numpy as np
import matplotlib

from dataset import fileutils
from features import cross_correlate

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import os.path


def plot_correlations(correlations, output, segment_start=None, segment_end=None, subplot_rows=1, subplot_cols = 6):
    pp = PdfPages(output)

    fig_mappings = dict()
    fig_files = defaultdict(list)
    pair_order = None
    for i, (f, corrs) in enumerate(sorted(correlations)):
        index = i % (subplot_cols*subplot_rows)
        if index == 0:
            fig, subplots = plt.subplots(subplot_rows, subplot_cols, sharey=True, squeeze=False)
            subplots = [ax for row in subplots for ax in row]
            for ax in subplots:
                ax.set_visible(False)
            fig_mappings[fig] = subplots

        fig_files[fig].append(f)
        corrmap = []

        if pair_order is None:
            pair_order = sorted(corrs.keys())
        for channel_pair in pair_order:
            frames = corrs[channel_pair]
            #Every channel pair becomes a row in the image
            corrdata = []
            window_sizes = set()
            for (window_start, window_end), time_deltas in sorted(frames.items()):
                if ((segment_start is not None and window_end <= segment_start) or
                    (segment_end is not None and window_start >= segment_end)):
                    continue
                (t_offset, correlation) = max(time_deltas, key=lambda x: x[1])
                corrdata.append(float(correlation))
                window_sizes.add(window_end - window_start)

            corrmap.append(corrdata)

        ax = subplots[index]
        ax.set_visible(True)
        ax.set_xticks([])
        ax.set_yticks([])
        heatmap = ax.imshow(corrmap)
        #ax.set_title("Cross correlations for channels from file: {}".format(f))
        if index == 0:
            ax.set_xlabel("Frames ({} s)".format(max(window_sizes)))
            ax.set_ylabel("Channel pairs")


    for fig, subplots in fig_mappings.items():
        title = ", ".join([os.path.basename(f) for f in sorted(fig_files[fig])])
        fig.suptitle(title, fontsize=8)
        #fig.colorbar(subplots.take(0))
        pp.savefig(fig)
        plt.close(fig)
    pp.close()


def class_corr_scatter(file_correlations, output, segment_start = None, segment_end=None):
    """Plots the data as a scatterplot with the classes interictal = 0, preictal = 1"""
    preictal_channel_corrs = defaultdict(list)
    interictal_channel_corrs = defaultdict(list)
    channels = set()
    for f, correlations in file_correlations:
        for channel_pair, frames in correlations.items():
            channels.add(channel_pair)
            for (window_start, window_end), time_corrlist in frames.items():
                corrlist = [correlation for (delta_time, correlation) in time_corrlist]
                if 'preictal' in f.lower():
                    preictal_channel_corrs[channel_pair].extend(corrlist)
                elif 'interictal' in f.lower():
                    interictal_channel_corrs[channel_pair].extend(corrlist)

    fig = plt.figure()
    interictal_data = [(i, interictal_channel_corrs[channel_pair]) for i, channel_pair in enumerate(channels)]
    preictal_data = [(i, preictal_channel_corrs[channel_pair]) for i, channel_pair in enumerate(channels)]

    interictal_x, interictal_y = zip(*interictal_data)
    preictal_x, preictal_y = zip(*preictal_data)
    plt.plot(interictal_x, interictal_y, color='blue', marker='x', linestyle='', label="Interictal", alpha=0.5)
    plt.plot(preictal_x, preictal_y, color='red', marker='o', linestyle='', label="Preictal", alpha=0.5)

    plt.xlabel("Channel pair")
    plt.ylabel("Correlation")
    plt.legend()
    fig.savefig(output)
    plt.close(fig)


def corr_box_plot(file_correlations, output, channels_per_plot=10, segment_start=None, segment_end=None):
    """Plots the correlations as a boxplot over the channels"""
    pp = PdfPages(output)

    preictal_channel_corrs = defaultdict(list)
    interictal_channel_corrs = defaultdict(list)
    channels = set()
    for f, correlations in file_correlations:
        for channel_pair, frames in correlations.items():
            channels.add(channel_pair)
            for (window_start, window_end), time_corrlist in frames.items():
                corrlist = [correlation for (delta_time, correlation) in time_corrlist]
                if 'preictal' in f.lower():
                    preictal_channel_corrs[channel_pair].extend(corrlist)
                elif 'interictal' in f.lower():
                    interictal_channel_corrs[channel_pair].extend(corrlist)


    channels = list(sorted(channels))

    for i in range(0, len(channels), channels_per_plot):
        plot_channels = channels[i:i+channels_per_plot]

        interictal_data = [interictal_channel_corrs[channel_pair] for channel_pair in plot_channels]
        preictal_data = [preictal_channel_corrs[channel_pair] for channel_pair in plot_channels]
        fig = plt.figure()

        interictal_bp = plt.boxplot(interictal_data, positions=[i*4 for i, c in enumerate(plot_channels)], sym='', vert=False)
        preictal_bp = plt.boxplot(preictal_data, positions=[i*4+1 for i,c in enumerate(plot_channels)], sym='', vert=False)

        for box in interictal_bp['boxes']:
            plt.setp(box, color='blue')

        for box in preictal_bp['boxes']:
            plt.setp(box, color='red')

        ticks, labels = zip(*[(4*i+2, "{}\n{}".format(*c)) for i, c in enumerate(plot_channels)])
        plt.yticks(ticks, labels, fontsize=10)
        plt.ylim(0, channels_per_plot*4+3)
        fig.subplots_adjust(left=0.3, right=0.95, top=0.90)
        # draw temporary red and blue lines and use them to create a legend

        blue_line = matplotlib.lines.Line2D([], [], color='blue', label='Interictal')
        red_line = matplotlib.lines.Line2D([], [], color='red', label='Preictal')
        plt.legend((blue_line, red_line), ("Interictal", "Preictal"), bbox_to_anchor=(0., 1.02, 1., .102),
                   loc=3, mode='expand', borderaxespad=0.0, ncol=2)

        plt.ylabel("Channel pairs")
        plt.xlabel("Correlation")
        pp.savefig(fig)
        plt.close(fig)

    #plt.legend()
    #fig.savefig(output)

    pp.close()


def get_delta_ts(file_correlations, segment_start=None, segment_end=None):
    """Extract the delta_t:s from the correlations"""
    delta_ts = defaultdict(list)
    for filename, correlations in file_correlations:
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
    for filename, correlations in file_correlations:
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


def boxplot_delta_t_distributions(file_correlations, output, channels_per_plot=1, **kwargs):
    pp = PdfPages(output)

    preictal_channel_corrs = defaultdict(lambda: defaultdict(list))
    interictal_channel_corrs = defaultdict(lambda: defaultdict(list))

    channels = set()
    for f, correlations in file_correlations:
        for channel_pair, frames in correlations.items():
            channels.add(channel_pair)
            for time_interval, time_corrlist in frames.items():
                for (delta_t, corr) in time_corrlist:
                    if 'preictal' in f.lower():
                        preictal_channel_corrs[channel_pair][delta_t].append(corr)
                    elif 'interictal' in f.lower():
                        interictal_channel_corrs[channel_pair][delta_t].append(corr)

   
    for i, channel_pair in enumerate(sorted(channels)):
        fig = plt.figure(dpi=300)

        interictal_times, interictal_data = zip(*sorted(interictal_channel_corrs[channel_pair].items()))
        preictal_times, preictal_data = zip(*sorted(preictal_channel_corrs[channel_pair].items()))

        interictal_bp = plt.boxplot(interictal_data, positions=[i*5-1 for i in range(len(interictal_times))], sym='')
        preictal_bp = plt.boxplot(preictal_data, positions=[i*5+1 for i in range(len(preictal_times))], sym='')

        for box in interictal_bp['boxes']:
            plt.setp(box, color='blue')

        for box in preictal_bp['boxes']:
            plt.setp(box, color='red')

        plt.xticks([5*i for i in range(len(interictal_times))], interictal_times, fontsize=10)
        plt.ylim(0,1)
        fig.subplots_adjust(left=0.3, right=0.95, top=0.90)

        # draw temporary red and blue lines and use them to create a legend
        blue_line = matplotlib.lines.Line2D([], [], color='blue', label='Interictal')
        red_line = matplotlib.lines.Line2D([], [], color='red', label='Preictal')
        plt.legend((blue_line, red_line), ("Interictal", "Preictal"), bbox_to_anchor=(0., 1.02, 1., .102),
                   loc=3, mode='expand', borderaxespad=0.0, ncol=2)

        plt.ylabel("Correlation")
        plt.xlabel("Time lag (s)")

        pp.savefig(fig, dpi=300, papertype='a0')
        plt.close(fig)
    pp.close()


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
    #test()
    #exit()
    #fileutils.process_segments(example_segments(), process_segment)
    #plot_welch_spectra(example_segments(), '../example.pdf')
    #exit(0)

    import argparse
    parser = argparse.ArgumentParser(description="Plots the cross correlation from previously created csv files.")

    parser.add_argument("--output", help="The output to plot to.", metavar="FILENAME")
    parser.add_argument("--segment-start", help="If this argument is supplied, only windows after this time in seconds will be used.", type=float, metavar="SECONDS")
    parser.add_argument("--segment-end", help="If this argument is supplied, only windows before this time will be used.", type=float, metavar="SECONDS")
    parser.add_argument("--mode", help="Which plot should be used.", choices=["heatmap", "class_scatter", "boxplot", "delta_times", "delta_dists", "boxdelta"])
    parser.add_argument("--channels-per-plot", help="How many channels should each plot have", type=int)
    parser.add_argument("csv_files", help="The csv files to be plotted, chan be files, directories or combinations thereof", nargs="+")
    #parser.add_argument("--channels", help="Selects a subset of the channels to use.")

    args = parser.parse_args()
    files = sorted(filter(lambda x: '.csv' in x, fileutils.expand_paths(args.csv_files)))

    correlations = cross_correlate.read_csv_files(files)

    if args.mode == "heatmap":
        plot_correlations(correlations, args.output, segment_start=args.segment_start, segment_end=args.segment_end)
    elif args.mode == "class_scatter":
        class_corr_scatter(correlations, args.output, segment_start=args.segment_start, segment_end=args.segment_end)
    elif args.mode == "boxplot":
        corr_box_plot(correlations, args.output, segment_start=args.segment_start, segment_end=args.segment_end)
    elif args.mode == 'delta_times':
        plot_delta_ts(correlations, output=args.output, segment_start=args.segment_start, segment_end=args.segment_end,
                  channels_per_plot=args.channels_per_plot)
    elif args.mode == 'delta_dists':
        plot_delta_t_distributions(correlations, output=args.output, segment_start=args.segment_start, segment_end=args.segment_end, channels_per_plot=args.channels_per_plot)
    elif args.mode == 'boxdelta':
        boxplot_delta_t_distributions(correlations, output=args.output, segment_start=args.segment_start, segment_end=args.segment_end, channels_per_plot=args.channels_per_plot)



