#!/usr/bin/env python
"""
Module for plotting cross correlations.
"""
import math
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import os.path
import csv

import fileutils
import segment
import cross_correlate


def plot_correlations(correlations, output, subplot_rows=1, subplot_cols = 6):
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
            corrdata, window_sizes = zip(*[(float(correlation), window_end - window_start) for (window_start, window_end), (t_offset, correlation) in sorted(frames.items())])
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
    plot_correlations(correlations, args.output)


