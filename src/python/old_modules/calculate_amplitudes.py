#!/usr/bin/env python

import math

import numpy as np

from dataset import fileutils, segment


def amplitude_histograms(s, bins=100):
    histograms = {channel : np.histogram(np.abs(s.get_channel_data(channel)), bins=bins) for channel in s.get_channels()}
    return  histograms


def process_segment(s, bins=100):
    """Calculates the amplitude of the segment. Returns the amplitudes as rows"""
    for channel, (counts, bins) in amplitude_histograms(s).items():
        header = ',\t'.join(["channel"] + ["bin_{}".format(bin) for bin in bins])
        yield header + "\n"
        data = ',\t'.join([channel] + [str(count) for count in counts])
        yield data + "\n"


def plot_amplitude_histograms(files, output, subplots_rows=1, subplots_cols=1, bins=100):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(output)

    #We assume the files have 16 channels
    channel_mappings = dict()
    n_figs = int(math.ceil(16/(subplots_rows*subplots_cols)))
    axes = []
    figs = []
    legends = dict()

    #Setup the axes for the files
    for i in range(n_figs):
        f, a = plt.subplots(subplots_rows, subplots_cols, squeeze=False)
        figs.append(f)
        axes.extend([ax for row in a for ax in row])

    for filename in files:
        s = segment.Segment(filename)
        for i, channel in enumerate(s.get_channels()):
            if channel not in channel_mappings:
                channel_mappings[channel] = i
                axes[channel_mappings[channel]].set_title(channel)
            ax = axes[channel_mappings[channel]]

            d, b, [line] = ax.hist(s.get_channel_data(channel), bins=bins, histtype='step', label=filename)
            if filename not in legends:
                legends[filename] = line

    for fig in figs:
        labels, lines = zip(*legends.items())
        fig.legend(lines, labels, loc='lower center')
        pp.savefig(fig, dpi=300, papertype='a0')

    pp.close()


def example_segments():
    segments = ['../data/Dog_1/Dog_1_preictal_segment_0001.mat', '../data/Dog_1/Dog_1_interictal_segment_0001.mat']
    return segments

if __name__ == '__main__':
    #fileutils.process_segments(example_segments(), process_segment)
    #plot_amplitude_histograms(example_segments(), '../example.pdf', bins=50)
    #exit(0)

    import argparse
    parser = argparse.ArgumentParser(description="Calculate the amplitude over the channels of the given segment files.")

    parser.add_argument("--bins", help="The number of bins to use in the amplitude histogram.", type=int, default=100)
    parser.add_argument("files", help="The files to process. This can either be the path to a matlab file holding the segment or a directory holding such files.", nargs='+')
    parser.add_argument("--plot", help="Plot the histograms to the given file name.", metavar="FILENAME")
    parser.add_argument("--write-csv", help="Writes the histograms to csv files.", action='store_true')
    args = parser.parse_args()

    files = sorted(fileutils.expand_paths(args.files))
    if args.write_csv:
        fileutils.process_segments(files, process_segment, output_format="{basename}_amplitude_histogram.csv")
    if args.plot:
        plot_amplitude_histograms(files, args.plot, bins=args.bins)


