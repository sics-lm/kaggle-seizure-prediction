#!/usr/bin/env python
"""
Module for plotting cross correlations.
"""
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import os.path

import fileutils
import cross_correlate


def plot_correlations(correlations, output, segment_start=None, segment_end=None, subplot_rows=1, subplot_cols = 6):
    pp = PdfPages(output)

    fig_mappings = dict()
    fig_files = defaultdict(list)
    pair_order = None
    for i, (f, corrs) in enumerate(sorted(correlations.items())):
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
    for f, correlations in file_correlations.items():
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
    for f, correlations in file_correlations.items():
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
    parser.add_argument("--mode", help="Which plot should be used.", choices=["heatmap", "class_scatter", "boxplot"])
    parser.add_argument("csv_files", help="The csv files to be plotted, chan be files, directories or combinations thereof", nargs="+")
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
    if args.mode == "heatmap":
        plot_correlations(correlations, args.output, segment_start=args.segment_start, segment_end=args.segment_end)
    elif args.mode == "class_scatter":
        class_corr_scatter(correlations, args.output, segment_start=args.segment_start, segment_end=args.segment_end)
    elif args.mode == "boxplot":
        corr_box_plot(correlations, args.output, segment_start=args.segment_start, segment_end=args.segment_end)


