"""
Module for dealing with combined features
"""
from __future__ import absolute_import
import pandas as pd

from . import correlation_convertion
from . import wavelet_classification


def load(segment_files, **kwargs):
    """Loads the multiple features from segment_files and concatenate them to a single dataframe. The feature loader to use is based on the file name. If the path contains 'wavelet', the wavelet feature loader will be used. If it contains 'corr' the cross-correlation feature loader will be used."""

    print("Loading files with kwargs: ", kwargs)
    dataframes = []
    for segment_file in segment_files:
        if 'wavelet' in segment_file:
            dataframes.append(wavelet_classification.load_csv(segment_file))
        elif 'corr' in segment_file:
            dataframes.append(correlation_convertion.load_and_pivot(segment_file))
        else:
            raise NotImplementedError("Don't know which feature loading function to use for {}.".format(segment_file))
    return pd.concat(dataframes, axis=1)

