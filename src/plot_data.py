#!/usr/bin/env python


import matplotlib.pyplot as plt
import segment
import os
import os.path

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


def plot_many(filenames, output_dir=None, format='png', columns_slice=None, channel_slice=None, **kwargs):
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
                output_path = "{}.{}".format(root, format)
            else:
                basename = os.path.basename(root)
                output_path = "{}.{}".format(os.path.join(output_dir, basename), format)
            fig.savefig(output_path, format=format, **kwargs)


def expand_paths(filenames, recursive=True):
    """Goes through the list of *filenames* and expands any directory to the files included in that directory.
    If *recursive* is True, any directories in the base directories will be expanded as well. If *recursive* is
    False, only normal files in the directories will be included.
    The returned list only includes non-directory files."""
    new_files = []
    for file in filenames:
        if os.path.isdir(file):
            if recursive:
                #We recurse over all files contained in the directory and add them to the list of files
                for dirpath, dirnames, subfilenames in os.walk(file):
                    new_files.extend([os.path.join(dirpath, fn) for fn in subfilenames])
            else:
                #No recursion, we just do a listfile on the files of any directoy in filenames
                for subfile in os.listdir(file):
                    if os.path.isfile(subfile):
                        new_files.append(os.path.join(file, subfile))
        elif os.path.isfile(file):
            new_files.append(file)
    return new_files


def test():
    s = segment.test_preictal()
    plot_segment(s, channel_slice=slice(0,1), column_slice=slice(0, 100))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Script for plotting segment files from the kaggle seizure classification competigion")
    parser.add_argument("-f", "--format", help="Format for the output files. This depends on the available matplotlib formats, usually 'pdf' and 'png' are available", default='png')
    parser.add_argument("-o", "--output", help="Output directory to use")
    parser.add_argument("files", help="The files to plot. This can either be the path to a matlab file holding the segment or a directory holding such files.", nargs='+')
    parser.add_argument("--channels", help="An interval of which channels to plot.", nargs=2, type=int)
    parser.add_argument("--sample", help="An interval of the data to plot.", nargs=2, type=int)


    args = parser.parse_args()
    files = expand_paths(args.files)
    channels = slice(*args.channels) if args.channels else None
    sample = slice(*args.sample) if args.sample else None
    plot_many(files, format=args.format, output_dir=args.output,
              channel_slice=channels, columns_slice=sample)
