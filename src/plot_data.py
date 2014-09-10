__author__ = 'erik'

import matplotlib.pyplot as plt
import segment

def plot_segment(segment_object, channel_slice=None, column_slice=None):
    """Plots the given segment. If *range* is given as a slice object, it will be used to select the data which will be plotted"""

    channels = segment_object.get_channels()
    if channel_slice is not None:
        channels = channels[channel_slice]

    n_channels = len(channels)
    data = segment_object.get_data()

    for i,channel in enumerate(channels):
        plot_number = i + 1  # Matplotlib starts the subplots at 1, not 0
        ax = plt.subplot(n_channels, 1, plot_number)
        if column_slice is None:
            ax.plot(data[i])
        else:
            channel_data = data[i][column_slice]
            ax.plot(channel_data)

def test():
    s = segment.test()
    plot_segment(s, channel_slice=slice(0,1), column_slice=slice(0, 100))

if __name__ == '__main__':
    test()