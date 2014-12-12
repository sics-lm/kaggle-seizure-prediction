from __future__ import absolute_import
from __future__ import print_function
from . import fileutils
from . import dataset

import pandas as pd
import numpy as np


def load_csv(filename, frame_length=12, sliding_frames=False):
    """
    Loads the wavelet of hills features from the given filename.
    :param filename: The filename to load features from.
    :param frame_length: The desired frame length in windows to use.
    :param sliding_frames: If True, the data will be extended by using sliding frames of the feature windows.
    :return: A DataFrame with the loaded features.
    """

    # Read the csv file with pandas and extract the values into an numpy array
    from_file_array = pd.read_table(filename, sep=',', dtype=np.float64, header=None).values

    # Assert that the csvfiles contain frames consisting 12 windows.
    assert_msg = 'file: "{}" does not have a column count divisible by 12 since it is: {}.'
    assert (from_file_array.shape[1] % 12) == 0, assert_msg.format(filename, from_file_array.shape[1])

    # Number of windows in the csv frame
    window_size = from_file_array.shape[1] / 12
    # Number of rows in the csv file
    n_rows = from_file_array.shape[0]*12

    # Reshaped array an array which is one windows wide and n_windows long.
    reshaped_array = from_file_array.reshape(n_rows, window_size)

    # Extract this function out into its own file and use it also with the cross correlation frames
    if sliding_frames:
        return pd.DataFrame(data=dataset.extend_data_with_sliding_frames(reshaped_array, frame_length))
    else:
        n_frames = reshaped_array.shape[0]/frame_length
        frame_size = window_size*frame_length
        return pd.DataFrame(data=reshaped_array.reshape(n_frames, frame_size))


def load_data_frames(feature_folder,
                     rebuild_data=False,
                     processes=4,
                     file_pattern="extract_features_for_segment.csv",
                     frame_length=12,
                     sliding_frames=False):
    return dataset.load_data_frames(feature_folder,
                                    load_function=load_csv,
                                    find_features_function=fileutils.find_feature_files,
                                    rebuild_data=rebuild_data,
                                    processes=processes,
                                    file_pattern=file_pattern,
                                    frame_length=frame_length,
                                    sliding_frames=sliding_frames)