import pandas as pd
import numpy as np
import glob
import os.path
import re
import multiprocessing
import random
from functools import partial

channel_pattern = re.compile(r'(?:[a-zA-Z0-9]*_)*(c[0-9]*|[A-Z]*_[0-9]*)$')

def convert_channel_name(name):
    """Pass"""
    match = re.match(channel_pattern, name)
    if match:
        return match.group(1) or match.group(2)
    else:
        return name


def load_and_pivot(filename, frame_length=1):
    """
    Loads the given csv. Will return a data frame with the channel
    pairs as columns. *frame_length* determines how many windows from
    the csv file will be collected into every row of the data
    frame.
    """
    with open(filename) as fp:
        dataframe = pd.read_csv(fp, sep="\t")
        channel_i = dataframe['channel_i'].map(convert_channel_name)
        channel_j = dataframe['channel_j'].map(convert_channel_name)
        dataframe['channels'] = channel_i.str.cat(channel_j, sep=":")
        pivoted = dataframe.pivot('start_sample', 'channels', 'correlation')

        if frame_length == 1:
            return pivoted
        else:
            df_length = len(pivoted)
            # Make sure the length of the frame is divisible by frame_length
            if df_length % frame_length != 0:
                raise(ValueError("The length {} of the dataframe in {} is not divisible by the frame length {}".format(df_length,
                                                                                                                       filename,
                                                                                                                       frame_lenght)))
            row_ranges = [np.arange(i, df_length, frame_length) for i in range(frame_length)]
            frames = [pivoted.iloc[row_range] for row_range in row_ranges]

            for frame in frames:
                frame.index = np.arange(df_length/frame_length)
            return pd.concat(frames, axis=1)


def load_correlation_files(feature_folder,
                           class_name,
                           file_pattern="5.0s.csv",
                           rebuild_data=False,
                           processes=1,
                           frame_length=1):
    cache_file = os.path.join(feature_folder, '{}_frame_length_{}_cache.pickle'.format(class_name, frame_length))

    if rebuild_data or not os.path.exists(cache_file):
        print("Rebuilding {} data".format(class_name))
        full_pattern="*{}*{}".format(class_name, file_pattern)
        glob_pattern=os.path.join(feature_folder, full_pattern)
        files=glob.glob(glob_pattern)
        segment_names = [os.path.basename(filename) for filename in files]
        if processes > 1:
            print("Reading files in parallel")
            pool = multiprocessing.Pool(processes)
            try:
                partial_load_and_pivot = partial(load_and_pivot, frame_length=frame_length)
                segment_frames = pool.map(partial_load_and_pivot, files)
            finally:
                pool.close()
        else:
            print("Reading files serially")
            segment_frames = [load_and_pivot(filename, frame_length=frame_length) for filename in files]

        complete_frame = pd.concat(segment_frames,
                                   names=('segment', 'start_sample'),
                                   keys=segment_names)

        complete_frame.to_pickle(cache_file)
    else:
        complete_frame = pd.read_pickle(cache_file)
    return complete_frame


def load_data_frames(feature_folder, rebuild_data=False,
                     processes=4, file_pattern="5.0s.csv",
                     frame_length=1):

    preictal = load_correlation_files(feature_folder,
                                      class_name="preictal",
                                      file_pattern=file_pattern,
                                      rebuild_data=rebuild_data,
                                      processes=processes,
                                      frame_length=frame_length)
    preictal['Preictal'] = 1

    interictal = load_correlation_files(feature_folder,
                                        class_name="interictal",
                                        file_pattern=file_pattern,
                                        rebuild_data=rebuild_data,
                                        processes=processes,
                                        frame_length=frame_length)
    interictal['Preictal'] = 0

    test = load_correlation_files(feature_folder,
                                  class_name="test",
                                  file_pattern=file_pattern,
                                  rebuild_data=rebuild_data,
                                  processes=processes,
                                  frame_length=frame_length)

    return interictal, preictal, test


def get_channel_df(dataframe):
    """Returns a dataframe with only the channel pairs as columns"""
    return dataframe.drop('Preictal', axis=1)


def get_class(dataframe):
    return dataframe['Preictal']


def get_segments(dataframe):
    """Returns the segment name part of the index for the dataframe"""
    return
