"""
Module for loading cross-correlation features.
"""

import re

import pandas as pd

from dataset import fileutils, dataset


channel_pattern = re.compile(r'(?:[a-zA-Z0-9]*_)*(c[0-9]*|[A-Z]*_[0-9]*)$')

def convert_channel_name(name):
    """Pass"""
    match = re.match(channel_pattern, name)
    if match:
        return match.group(1) or match.group(2)
    else:
        return name


def old_load_and_pivot(dataframe):
    """Old version of load and pivot which uses the old, redundant version where channel_i and channel_j are columns"""
    channel_i = dataframe['channel_i'].map(convert_channel_name)
    channel_j = dataframe['channel_j'].map(convert_channel_name)
    dataframe['channels'] = channel_i.str.cat(channel_j, sep=":")

    dataframe.drop(['channel_i', 'channel_j', 'end_sample', 't_offset'], axis=1, inplace=True)
    max_corrs = dataframe.groupby(['channels', 'start_sample'], as_index=False).max()
    pivoted = max_corrs.pivot('start_sample', 'channels', 'correlation')
    return pivoted


def new_load_and_pivot(dataframe):
    """New version which assumes the columns where the channel pairs are already columns"""
    dataframe.drop(['end_sample', 't_offset'], axis=1, inplace=True)
    max_corrs = dataframe.groupby('start_sample').max()
    return max_corrs


def load_and_pivot(filename, frame_length=1, sliding_frames=True):
    """
    Loads the given csv. Will return a data frame with the channel
    pairs as columns. *frame_length* determines how many windows from
    the csv file will be collected into every row of the data
    frame.
    """
    with open(filename) as fp:
        dataframe = pd.read_csv(fp, sep="\t")

        #Figure out if this file contains the old or new format
        if 'channel_i' in dataframe.columns:
            pivoted = old_load_and_pivot(dataframe)
        else:
            pivoted = new_load_and_pivot(dataframe)

        if frame_length == 1:
            return pivoted
        else:
            if sliding_frames:
                return dataset.create_sliding_frames(pivoted, frame_length=frame_length)
            else:
                return dataset.reshape_frames(pivoted, frame_length=frame_length)


def load_data_frames(feature_folder,
                     **kwargs):
    """
    Loads the dataframes for the feature files in *feature_folder*. This function has been deprecated in favor of dataset.load_data_frames.
    """
    return dataset.load_data_frames(feature_folder,
                                    load_function=load_and_pivot,
                                    find_features_function=fileutils.find_feature_files,
                                    **kwargs)
