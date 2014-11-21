#!/usr/bin/env python
from __future__ import absolute_import

import os
import os.path

import matplotlib.pyplot as plt

from ..datasets import fileutils, segment


def plot_segment(segment_object, channel_slice=None, column_slice=None):
    """Plots the given segment. If the argument *channel_slice* is given as a slice object,
    only the sliced channels (rows in the segment data matrix) will be plotted.
    If *column_slice* is given as a slice object,
    it will be used to select which columns of the electrode data which will be plotted."""

    channels = segment_object.get_channels()
    if channel_slice is not None:
        channels = channels[channel_slice]

    n_channels = len(channels)
    data = segment_object.get_data()
    fig, subplots = plt.subplots(n_channels, 1, sharex=True, sharey=True)

    for i, axes in enumerate(subplots):
        if column_slice is None:
            axes.plot(data[i])
        else:
            channel_data = data[i][column_slice]
            axes.plot(channel_data)

    plt.suptitle(segment_object.get_filename())
    return fig, subplots


def plot_many(filenames, output_dir=None, output_format='png', columns_slice=None, channel_slice=None, **kwargs):
    """Plots all segment files in the given directory and outputs the plot to an image with the same name.
    The keyword arguments *columns_slice* and *channel_slice* has the same meaning as for plot_segment.
    *output_dir* can be set to a directory to write the plots to, if None is given, the same directory as
    the input data will be used.
    *format* determines the output format of the images, the default is PNG. Additional keyword
    arguments are sent to matplotlib.pyplot.savefig"""
    for filename in filenames:
        root, ext = os.path.splitext(filename)
        if '.mat' == ext.lower():
            seg = segment.Segment(filename)
            fig, subplots = plot_segment(seg, column_slice=columns_slice, channel_slice=channel_slice)
            if output_dir is None:
                output_path = "{}.{}".format(root, output_format)
            else:
                basename = os.path.basename(root)
                output_path = "{}.{}".format(os.path.join(output_dir, basename), output_format)
            fig.savefig(output_path, format=output_format, **kwargs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Script for plotting segment files from the kaggle seizure classification competigion")
    parser.add_argument("-f", "--format", help="Format for the output files. This depends on the available matplotlib formats, usually 'pdf' and 'png' are available", default='png')
    parser.add_argument("-o", "--output", help="Output directory to use")
    parser.add_argument("files", help="The files to plot. This can either be the path to a matlab file holding the segment or a directory holding such files.", nargs='+')
    parser.add_argument("--channels", help="An interval of which channels to plot.", nargs=2, type=int)
    parser.add_argument("--sample", help="An interval of the data to plot.", nargs=2, type=int)


    args = parser.parse_args()
    files = fileutils.expand_paths(args.files)
    channels = slice(*args.channels) if args.channels else None
    sample = slice(*args.sample) if args.sample else None
    plot_many(files, output_format=args.format, output_dir=args.output,
              channel_slice=channels, columns_slice=sample)
