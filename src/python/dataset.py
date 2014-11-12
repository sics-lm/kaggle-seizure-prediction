"""Python module for manipulating datasets."""
import random
import os
import os.path
import logging
import glob
import multiprocessing
from functools import partial
import logging

import pandas as pd
import numpy as np
import sklearn
from sklearn import cross_validation

import fileutils
from features_combined import load as load_combined
import basic_segment_statistics

def first(iterable):
    """Returns the first element of an iterable"""
    for element in iterable:
        return element

class SegmentCrossValidator:
    """Wrapper for the scikit_learn CV generators to generate folds on a segment basis."""
    def __init__(self, dataframe, base_cv=None, **cv_kwargs):
        # We create a copy of the dataframe with a new last level
        # index which is an enumeration of the rows (like proper indices)
        self.all_segments = pd.DataFrame({'Preictal': dataframe['Preictal'], 'i': np.arange(len(dataframe))})
        self.all_segments.set_index('i', append=True, inplace=True)

        #Now create a series with only the segments as rows. This is what we will pass into the wrapped cross validation generator
        self.segments = self.all_segments['Preictal'].groupby(level='segment').first()
        self.segments.sort(inplace=True)

        if base_cv is None:
            self.cv = cross_validation.StratifiedKFold(self.segments, **cv_kwargs)
        else:
            self.cv = base_cv(self.segments, **cv_kwargs)

    def __iter__(self):
        """
        Return a generator object which returns a pair of indices for every iteration.
        """
        for training_indices, test_indices in self.cv:
            #The indices returned from self.cv are relative to the segment name data series, we pick out the segment names they belong to
            training_segments = list(self.segments[training_indices].index)
            test_segments = list(self.segments[test_indices].index)
            # Now that we have the segment names, we pick out the rows in the properly indexed dataframe
            all_training_indices = self.all_segments.loc[training_segments]
            all_test_indices = self.all_segments.loc[test_segments]

            #Now pick out the values for only the 'i' level index of the rows which matched the segment names
            original_df_training_indices = all_training_indices.index.get_level_values('i')
            original_df_test_indices = all_test_indices.index.get_level_values('i')

            yield original_df_training_indices, original_df_test_indices


    def __len__(self):
        return len(self.cv)


def mean(*dataframes):
    """Returns the means of the given dataframe, calculated without
    concatenating the frame"""
    lengths = sum([len(dataframe) for dataframe in dataframes])
    sums = dataframes[0].sum()
    for dataframe in dataframes[1:]:
        sums += dataframe.sum()
    means = sums / lengths
    return means


def scale(dataframes, center=True, scale=True, inplace=False):
    """Returns standardized (mean 0, standard deviation 1) versions of the given data frames.
    Args:
        dataframes: A variable number of inplace arguments which should be the
                    dataframes to standardize.
        center: If True, the columns of the frame will have mean 0.
        scale:  If True, the columns will have standard deviation of 1.
        inplace: If True, the dataframes will be standardized inplace, and
        no new dataframe is created.
    Returns:
        A list of the standardized dataframes. If inplace=True, it will be the
        same dataframes as the argument. If inplace=False, new dataframes will
        have been created.
    """
    ## This can be quite memory intensive, especially if the
    ## dataframes are big
    complete_frame = pd.concat(dataframes, axis=0)
    mean = complete_frame.mean()
    std = complete_frame.std()
    mean['Preictal'] = 0  # Don't screw up the class label
    std['Preictal'] = 1  # Don't change the class label

    if inplace:
        for dataframe in dataframes:
            if center:
                dataframe -= mean
            if scale:
                dataframe /= std
        return dataframes
    else:
        new_dataframes = []
        for dataframe in dataframes:
            new_dataframe = dataframe
            if center:
                new_dataframe = new_dataframe - mean
            if scale:
                new_dataframe = new_dataframe / std
            new_dataframes.append(new_dataframe)
        return new_dataframes


def split_experiment_data(interictal,
                          preictal,
                          training_ratio, do_downsample=True,
                          downsample_ratio=2.0,
                          do_segment_split=True,
                          random_state=None):
    """
    Creates a split of the data into two seperate data frames.
    Args:
        *interictal*: A data frame containing the interictal samples.
        *preictal*: A data frame containing the preictal samples.
        *training_ratio*: a value in the range (0,1) which indicates the ratio
                          of samples to use for training.
        *do_downsample*: flag of whether to down sample the larger class.
        *downsample_ratio*: The maximum imbalance ratio to use for down sampling.
        *do_segment_split*: flag of whether to split based on segment names.
    Returns:
        A partition of the concatenated interictal and preictal data frames into
        two seperate and disjunct data frames, such that the first partition
        contains a ratio of *training_ratio* of all the data.
    """
    dataset = merge_interictal_preictal(interictal, preictal,
                                        do_downsample=do_downsample,
                                        downsample_ratio=downsample_ratio,
                                        do_segment_split=do_segment_split,
                                        random_state=random_state)
    return split_dataset(dataset,
                         training_ratio=training_ratio,
                         do_segment_split=do_segment_split,
                         random_state=random_state)



def merge_interictal_preictal(interictal, preictal,
                              do_downsample=True,
                              downsample_ratio=2.0,
                              do_segment_split=True,
                              random_state=None):
    """
    Merges the interictal and preictal data frames to a single data frame. Also sorts the multilevel index.

    Args:
        *interictal*: A data frame containing the interictal samples.
        *preictal*: A data frame containing the preictal samples.
        *do_downsample*: flag of whether to down sample the larger class.
        *downsample_ratio*: The maximum imbalance ratio to use for down sampling.
        *do_segment_split*: flag of whether to split based on segment names.
    Returns:
        A data frame containing both interictal and preictal data. The multilevel index of the data frame is sorted.
    """
    logging.info("Merging interictal and preictal datasets")
    try:
        preictal.sortlevel('segment', inplace=True)
        if isinstance(preictal.columns, pd.MultiIndex):
            preictal.sortlevel(axis=1, inplace=True)

        interictal.sortlevel('segment', inplace=True)
        if isinstance(interictal.columns, pd.MultiIndex):
            interictal.sortlevel(axis=1, inplace=True)
    except TypeError:
        logging.warn("TypeError when trying to merge interictal and preictal sets.")

    if do_downsample:
        logging.info("Downsampling datasets")
        interictal = downsample(interictal, len(preictal) * downsample_ratio,
                                do_segment_split=do_segment_split,
                                random_state=random_state)
    dataset = pd.concat((interictal, preictal))
    dataset.sortlevel('segment', inplace=True)
    return dataset


def downsample(df1, n_samples, do_segment_split=True, random_state=None):
    """
    Returns a downsampled version of *df1* so that it contains at most
    a ratio of *downsample_ratio* samples of df2.
    Args:
        *df1*: The dataframe which should be downsampled.
        *n_samples*: The number of samples to include in the sample.
        *do_segment_split*: Whether the downsampling should be done per segment.
    Returns:
        A slice of df1 containing len(df2)*downsample_ratio number of samples.
    """
    if random_state is not None:
        random.seed(random_state)

    if do_segment_split:
        df1_segments = list(sorted(df1.index.get_level_values('segment').unique()))
        samples_per_segment = len(df1)/len(df1_segments)

        n_segment_samples = int(n_samples / samples_per_segment)
        if n_segment_samples < len(df1_segments):
            sample_segments = random.sample(df1_segments, n_segment_samples)

            return df1.loc[sample_segments]
        else:
            return df1

    else:
        if n_samples < len(df1):
            print('N_samples: {}'.format(n_samples))
            sample_indices = random.sample(range(len(df1)), n_samples)
            return df1.iloc[sample_indices]
        else:
            print('N_samples: {}'.format(len(df1)))
            return df1


def split_dataset(dataframe, training_ratio=.8, do_segment_split=True, shuffle=False, random_state=None):
    """
    Splits the dataset into a training and test partition.
    Args:
        *dataframe*: A data frame to split. Should have a 'Preictal' column.
        *training_ratio*: The ratio of the data to use for the first part.
        *do_segment_split*: If True, the split will be done on whole segments.
        *shuffle*: If true, the split will shuffle the data before splitting.
    Returns:
        A pair of disjunct data frames, where the first frame contain
        *training_ratio* of all the data.
    """
    ## We'll make the splits based on the sklearn cross validators,
    ## We calculate the number of folds which correspond to the
    ## desired training ratio. If *r* is the training ratio and *k*
    ## the nubmer of folds, we'd like *r* = (*k* - 1)/*k*, that is,
    ## the ratio should be the same as all the included folds divided
    ## by the total number of folds. This gives us *k* = 1/(1-*r*)
    k = int(np.floor(1/(1 - training_ratio)))

    if do_segment_split:
        # We use the segment based cross validator to get a stratified split.
        cv = SegmentCrossValidator(dataframe,
                                   n_folds=k,
                                   shuffle=shuffle,
                                   random_state=random_state)
    else:
        # Don't split by segment, but still do a stratified split
        cv = cross_validation.StratifiedKFold(dataframe['Preictal'],
                                              n_folds=k,
                                              shuffle=shuffle,
                                              random_state=random_state)

    training_indices, test_indices = first(cv)
    return dataframe.iloc[training_indices], dataframe.iloc[test_indices]


def test_k_fold_segment_split():
    interictal_classes = np.zeros(120)
    preictal_classes = np.ones(120)
    classes = np.concatenate((interictal_classes, preictal_classes,))
    segments = np.arange(12)
    i = np.arange(240)

    index = pd.MultiIndex.from_product([segments, np.arange(20)], names=('segment', 'start_sample'))

    dataframe = pd.DataFrame({'Preictal': classes, 'i': i}, index=index)

    # With a 6-fold cross validator, we expect each held-out fold to contain exactly 2 segments, one from each class
    cv1 = SegmentCrossValidator(dataframe, n_folds=6, shuffle=True, random_state=42)
    cv2 = SegmentCrossValidator(dataframe, n_folds=6, shuffle=True, random_state=42)

    for (training_fold1, test_fold1), (training_fold2, test_fold2) in zip(cv1, cv2):
        assert np.all(training_fold1 == training_fold1) and np.all(test_fold1 == test_fold2)



def combine_features(dataframes, labeled=True):
    """
    Combine the features of the dataframes by segment. The dataframe needs
    to have a 'segment' level in their index, and the innermost index needs
    to have the same number of rows per segment.
    """

    for dataframe in dataframes:
        normalize_segment_names(dataframe, inplace=True)
    if labeled:
        combined_dataframes = pd.concat([df.drop('Preictal', axis=1)
                                         for df in dataframes],
                                        axis=1)
        combined_dataframes['Preictal'] = dataframes[0]['Preictal']
    else:
        combined_dataframes = pd.concat(dataframes,
                                        axis=1)

    combined_dataframes.sortlevel('segment', inplace=True)
    return combined_dataframes


def normalize_segment_names(dataframe, inplace=False):
    """
    Makes the 'segment' index of the dataframe have names which correspond to the original matlab segment names.
    """
    index_values = dataframe.index.get_values()
    fixed_values = [(fileutils.get_segment_name(filename), frame) for filename, frame in index_values]
    if not inplace:
        dataframe = dataframe.copy()
    dataframe.index = pd.MultiIndex.from_tuples(fixed_values, names=dataframe.index.names)
    return dataframe


def load_data_frames(feature_folder,
                     classes=('interictal', 'preictal', 'test'),
                     sliding_frames=False,
                     segment_statistics=False,
                     **kwargs):
    """
                     load_function=None,
                     rebuild_data=False,
                     processes=4,
                     file_pattern="5.0s.csv",
                     frame_length=1,
                     sliding_frames=True):
    """
    if 'preictal' in classes:
        preictal = load_feature_files(feature_folder,
                                      class_name="preictal",
                                      sliding_frames=sliding_frames,
                                      **kwargs)
        preictal['Preictal'] = 1
    else:
        preictal = pd.DataFrame(np.zeros(0))
    if 'interictal' in classes:
        interictal = load_feature_files(feature_folder,
                                        class_name="interictal",
                                        sliding_frames=sliding_frames,
                                        **kwargs)
        interictal['Preictal'] = 0
    else:
        interictal = pd.DataFrame(np.zeros(0))

    if 'test' in classes:
        test = load_feature_files(feature_folder,
                                  class_name="test",
                                  # Never use sliding frames for the test-data
                                  sliding_frames=False,
                                  **kwargs)
    else:
        test = pd.DataFrame(np.zeros(0))

    if segment_statistics:
        try:
            logging.info("Loading segment statistics file from {}".format(feature_folder))
            segment_statistics = basic_segment_statistics.read_folder(feature_folder)
        except FileNotFoundError:
            logging.warning("There is not statistics file in {}".format(feature_folder))
        interictal = interictal.join(segment_statistics)
        preictal = preictal.join(segment_statistics)
        test = test.join(segment_statistics)

    preictal.sortlevel('segment', inplace=True)
    if isinstance(preictal.columns, pd.MultiIndex):
        preictal.sortlevel(axis=1, inplace=True)

    interictal.sortlevel('segment', inplace=True)
    if isinstance(interictal.columns, pd.MultiIndex):
        interictal.sortlevel(axis=1, inplace=True)

    test.sortlevel('segment', inplace=True)
    if isinstance(test.columns, pd.MultiIndex):
        test.sortlevel(axis=1, inplace=True)

    return interictal, preictal, test


def load_feature_files(feature_folder,
                       class_name,
                       load_function=load_combined,
                       find_features_function=fileutils.find_grouped_feature_files,
                       rebuild_data=False,
                       frame_length=12,
                       sliding_frames=False,
                       processes=1,
                       output_folder=None,
                       file_pattern="*segment*.csv"):
    """
    Loads all the files matching the class name and patter from the given feature folder.
    :param feature_folder: A folder containing files to load.
    :param class_name: The name of the class to load, 'interictal', 'preictal' or 'test'.
    :param load_function: A function which given a feature file or list of files should return a dataframe for that
                          feature.
    :param find_features_function: A function which takes a folder or list of folders and returns a list of
                                   dictionaries. The dictionaries should have the keys 'segment' and 'files'. The
                                   values of 'files' will be sent to the load function to load the features and the
                                   value of 'segment' will be used to tag the dataframe.
    :param rebuild_data: If True, the data will be rebuilt, even if a data cache file exists.
    :param frame_length: The desired length in windows of the features.
    :param sliding_frames: If True, the features will be oversampled by using a sliding window approach.
    :param processes: The number of processes to use for parallel loading of feature files.
    :param output_folder: The file to save the concatenated feature data frame caches to.
    :param file_pattern: A pattern wich will be used to select what files to load as features.
    :return: A pandas dataframe where all the features loaded from feature folder with the given class are
             concatenated. The index will have a level called 'segment' with the segment name for the feature frames.
    """
    cache_file_basename = fileutils.generate_filename('cache',
                                                      '.pickle',
                                                      [class_name,
                                                       'frame_length_{}'.format(frame_length)],
                                                      dict(sliding_frames=sliding_frames))
    if output_folder is None:
        output_folder = feature_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cache_file = os.path.join(output_folder, cache_file_basename)

    if rebuild_data or not os.path.exists(cache_file):
        logging.info("Rebuilding {} data from {}".format(class_name, feature_folder))

        feature_files = find_features_function(feature_folder,
                                               class_name=class_name,
                                               file_pattern=file_pattern)
        complete_frame = rebuild_features(feature_files,
                                          class_name,
                                          load_function,
                                          frame_length=frame_length,
                                          sliding_frames=sliding_frames,
                                          processes=processes)
        complete_frame.to_pickle(cache_file)
    else:
        logging.info("Loading {} data from "
                     "cache file {}".format(class_name,
                                            cache_file))
        complete_frame = pd.read_pickle(cache_file)

    return complete_frame


def rebuild_features(feature_file_dicts,
                     class_name,
                     load_function,
                     processes=1,
                     frame_length=1,
                     sliding_frames=False):
    """
    Loads all the features from the given feature folder matching the class name and file pattern. It combines them in a pandas dataframe and assigns a multilevel index to the rows based on the filenames the feature rows are taken from.
    Args:
        feature_file_dicts: A list of dictioneries, where each inner dictionary should have the keys 'segment' and 'files'. Segment should be the name of the segment which is loaded, while 'files' should be the argument to *load_function*, typically a single or multiple filenames.
        class_name: The class name of the files to load, typically {'interictal', 'preictal', 'test'}.
        load_function: Function to use for loading the feature files.
        file_pattern: A glob pattern used to select the matching files in the feature folder. Can be used to selectivily load files.
        rebuild_data: If False, a cached version of the data will be loaded if there is one. If True, the data will always be rebuilt, wich replaces any cached version already present.
        processes: The number of processes to use for reading the data files in parallel.
        frame_length: The length in number of windows each feature vector should be.
        sliding_frames: If True, the feature-vectors will be produced by a sliding frame over all the windows of each feature file.
    Return:
        A pandas DataFrame with the feature frames. The frame will have a MultiIndex with the original matlab segment names and the frame number of the feature frames.
    """
    tupled = [(feature['segment'], feature['files']) for feature in feature_file_dicts]
    segment_names, feature_files = zip(*tupled)

    if processes > 1:
        segment_frames = load_files_parallel(feature_files,
                                             load_function=load_function,
                                             processes=processes,
                                             frame_length=frame_length,
                                             sliding_frames=sliding_frames)
    else:
        segment_frames = load_files_serial(feature_files,
                                           load_function=load_function,
                                           frame_length=frame_length,
                                           sliding_frames=sliding_frames)

    complete_frame = pd.concat(segment_frames,
                               names=('segment', 'frame'),
                               keys=segment_names)

    complete_frame.sortlevel('segment', inplace=True)
    if np.count_nonzero(np.isnan(complete_frame)) != 0:
        logging.warning("NaN values found, using interpolation")
        complete_frame = complete_frame.interpolate(method='linear')

    return complete_frame


def load_files_parallel(feature_files, load_function, processes, **kwargs):
    logging.info("Reading files in parallel")
    pool = multiprocessing.Pool(processes)
    try:
        partial_load_and_pivot = partial(load_function, **kwargs)
        segment_frames = pool.map(partial_load_and_pivot, feature_files)
    finally:
        pool.close()
    return segment_frames


def load_files_serial(feature_files, load_function, **kwargs):
    logging.info("Reading files serially")
    return [load_function(files, **kwargs)
            for files in feature_files]


def reshape_frames(dataframe, frame_length=12):
    """
    Returns a new dataframe with the given frame length.
    Args:
        dataframe: A pandas DataFrame with one window per row.
        frame_length: The desired number of windows for each feature frame. Must divide the number of windows in *dataframe* evenly.
    Returns:
        A new pandas DataFrame with the desired window frame width. The columns of the new data-frame will be multi-index so that
        future concatenation of data frames align properly.
    """
    # Assert that the length of the data frame is divisible by
    # frame_length
    n_windows, window_width = dataframe.shape

    if n_windows % frame_length != 0:
        raise ValueError("The dataframe has {} windows which"
                         " is not divisible by the frame"
                         " length {}".format(n_windows, frame_length))
    values = dataframe.values
    n_frames = n_windows / frame_length
    frame_width = window_width * frame_length
    window_columns = dataframe.columns
    column_index = pd.MultiIndex.from_product([range(frame_length),
                                               window_columns],
                                              names=['window', 'feature'])
    reshaped_frame = pd.DataFrame(data=values.reshape(n_frames,
                                                      frame_width),
                                  columns=column_index)
    reshaped_frame.sortlevel(axis=1)
    return reshaped_frame


def create_sliding_frames(dataframe, frame_length=12):
    """ Wrapper for the extend_data_with_sliding_frames function wich works with numpy arrays. This version does the data-frame conversion for us.
    """
    extended_array = extend_data_with_sliding_frames(dataframe.values)
    # We should preserve the columns of the dataframe, otherwise
    # concatenating different dataframes along the row-axis will give
    # wrong results
    window_columns = dataframe.columns
    column_index = pd.MultiIndex.from_product([range(frame_length),
                                               window_columns],
                                              names=['window', 'feature'])
    return pd.DataFrame(data=extended_array,
                        columns=column_index)


def extend_data_with_sliding_frames(source_array, frame_length=12):
    """
    Creates an array of frames from the given array of windows using a sliding window.
    Args:
        source_array: a numpy array with the shape (n_windows, window_length)
        frame_length: The desired window length of the frames.
    """
    n_rows = source_array.shape[0]
    window_size = source_array.shape[1]

    #Number of frames that we can generate
    n_sliding_frames = n_rows-(frame_length-1)
    #The column size of our new frames
    frame_size = window_size*frame_length

    dest_array = np.zeros((n_sliding_frames, frame_size), dtype=np.float64)

    for i in range(0, n_sliding_frames):
        dest_array[i] = source_array[i:i+frame_length].reshape(1, frame_size)

    return dest_array


if __name__ == '__main__':
    test_k_fold_segment_split()
