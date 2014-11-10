"""
Module for dealing with combined features
"""
import pandas as pd



def load(segment_files, **kwargs):
    """Loads the multiple features from segment_files and concatenate them to a single dataframe. The feature loader to use is based on the file name. If the path contains 'wavelet', the wavelet feature loader will be used. If it contains 'corr' the cross-correlation feature loader will be used."""

    print("Loading files with kwargs: ", kwargs)
    dataframes = []
    for segment_file in segment_files:
        if 'wavelet' in segment_file:
            import wavelet_classification
            dataframes.append(wavelet_classification.load_csv(segment_file))
        elif 'corr' in segment_file:
            import correlation_convertion
            dataframes.append(correlation_convertion.load_and_pivot(segment_file))
        else:
            raise NotImplementedError("Don't know which feature loading function to use for {}.".format(segment_file))
    return pd.concat(dataframes, axis=1)
