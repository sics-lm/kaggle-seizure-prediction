#!/usr/bin/env python
"""
Module for producing spectra using the Welch method as provided by scipy.
"""
import numpy as np
import math
import scipy.signal
from collections import defaultdict
import matplotlib.pyplot as plt

import fileutils
import segment


def get_channel_power_variance(files):
    channel_dict = group_channel_spectra(files)
    variances = process_frequency(channel_dict, power_variance)
    return variances


def power_variance(frequency_mappings):
    """Calculates the variance over power for each frequncy in the list of frequency_mappings"""
    frequency_lists = defaultdict(list)
    for mapping in frequency_mappings:
        for f,p in mapping.items():
            frequency_lists[f].append(p)
    return { f : np.var(ps) for f,ps in frequency_lists.items()}


def process_frequency(channel_dict, process_fun):
    """
    Applies the function *process_fun* to the list of frequency : power mappings in channel dict. Returns a dictionary
    with channels to the result from process_fun
    """
    return { channel : process_fun(freqs) for channel, freqs in channel_dict.items()}


def group_channel_spectra(files):
    """Groups the spectras from the given files by their channels. The groups per channel are lists of dictionaries mapping frequency to power."""
    channel_dict = defaultdict(list)
    for f in files:
        channel_spectra = get_welch_spectra(segment.Segment(f))
        for channel, (f, pwelch_spec) in channel_spectra.items():
            channel_dict[channel].append({ freq : pow for freq, pow in zip(f, pwelch_spec)})
    #channel_dict now contains the frequencies and pwelch powers
    return channel_dict


def get_welch_spectra(s):
    freq = s.get_sampling_frequency()
    spectra = { channel : scipy.signal.welch(s.get_channel_data(channel).astype('float32'), freq) for channel in s.get_channels()}
    return spectra

# def process_segment(s, bins=100):
#     """Calculates the amplitude of the segment. Returns the amplitudes as rows"""
#     for channel, (counts, bins) in amplitude_histograms(s).items():
#         header = ',\t'.join(["channel"] + ["bin_{}".format(bin) for bin in bins])
#         yield header + "\n"
#         data = ',\t'.join([channel] + [str(count) for count in counts])
#         yield data + "\n"
#

def plot_welch_spectra(files, output, subplots_rows=1, subplots_cols=1, bins=100):
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
        spectra = get_welch_spectra(s)
        for i, (channel, (f, pwelch_spec))  in enumerate(spectra.items()):
            if channel not in channel_mappings:
                channel_mappings[channel] = i
                ax = axes[channel_mappings[channel]]
                ax.set_title(channel)
                ax.set_xlabel('Hz')
                ax.set_ylabel('PSD')
            ax = axes[channel_mappings[channel]]

            [line] = ax.plot(f, pwelch_spec, label=filename)
            if filename not in legends:
                legends[filename] = line

    for fig in figs:
        labels, lines = zip(*legends.items())
        fig.legend(lines, labels, loc='lower center')
        pp.savefig(fig, dpi=300, papertype='a0')

    pp.close()


def plot_power_variance(power_variances):
    """
    Creates plots for the power variances in the dictionary *power_variances* as obtained by get_channel_power_variance.
    *sb_rows* and *sb_cols* decide the number of subplot rows and columns.
    """
    n_channels = len(power_variances)
    fig = plt.figure()

    for channel, variances in power_variances.items():
        frequencies, powers = zip(*list(sorted(variances.items())))
        plt.plot(frequencies, powers, label=channel)

    plt.xlabel("Hz")
    plt.ylabel("Var(PSD)")
    plt.suptitle("Variance over frequency power over segments")
    plt.legend()

    return fig



def example_segments():
    segments = ['../data/Dog_1/Dog_1_preictal_segment_0001.mat', '../data/Dog_1/Dog_1_interictal_segment_0001.mat']
    return segments

if __name__ == '__main__':
    #fileutils.process_segments(example_segments(), process_segment)
    #plot_welch_spectra(example_segments(), '../example.pdf')
    #exit(0)

    import argparse
    parser = argparse.ArgumentParser(description="Plots the spectra of files using the Welch method as provided by scipy.")

    parser.add_argument("files", help="The files to process. This can either be the path to a matlab file holding the segment or a directory holding such files.", nargs='+')
    parser.add_argument("-o", "--output", help="Plot the histograms to the given file name.", metavar="FILENAME", dest='output')
    parser.add_argument("--method", help="The method to use for creating the spectra")
    #parser.add_argument("--write-csv", help="Writes the histograms to csv files.", action='store_true')
    args = parser.parse_args()

    files = sorted(fileutils.expand_paths(args.files))
    # if args.write_csv:
    #     fileutils.process_segments(files, process_segment, output_format="{basename}_amplitude_histogram.csv")
    if args.output:
        plot_welch_spectra(files, args.output)


