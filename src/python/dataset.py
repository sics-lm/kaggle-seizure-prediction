"""Python module for manipulating datasets."""
import random

import pandas as pd
import numpy as np
import sklearn
from sklearn import cross_validation


class SegmentCrossValidator:
    """Wrapper for the scikit_learn CV generators to generate folds on a segment basis."""
    def __init__(self, dataframe, base_cv=None, **cv_kwargs):
        # We create a copy of the dataframe with a new last level index which is an enumeration of the rows (like proper indices)
        self.dataframe = dataframe.reset_index('start_sample', drop=True)
        self.dataframe['i'] = np.arange(len(dataframe))
        self.dataframe.set_index('i', append=True, inplace=True)

        #We create a new series with only the class label
        self.all_segments = self.dataframe['Preictal']

        #Now create a series with only the segments as rows. This is what we will pass into the wrapped cross validation generator
        self.segments = self.all_segments.groupby(level='segment').first()

        if base_cv == None:
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

            # Now that we have the segment names, we pick out the rows in the properly indexed
            all_training_indices = self.all_segments.loc[training_segments]
            all_test_indices = self.all_segments.loc[test_segments]

            #Now pick out the values for only the 'i' level index of the rows which matched the segment names
            original_df_training_indices = all_training_indices.index.get_level_values('i')
            original_df_test_indices = all_test_indices.index.get_level_values('i')

            yield original_df_training_indices, original_df_test_indices


    def __len__(self):
        return len(self.cv)


def split_segment_names(dataframe, split_ratio):
    """
    Returns a pair of disjunct lists of segment names, such that the
    first set contains *split_ratio* of the total segment names.
    Args:
        *dataframe*: a pandas dataframe where the index has a level called
                     'segment' which the split will be done over.
        *split_ratio*: a value in the interval (0,1) of the ratio of segment
                       names to put in the first set.
    Returns:
        A pair of sorted names lists, where the first contains a ratio of s
        egment names equal to *split_ratio* times the total number of segments.
    """
    segment_names = set(dataframe.index.get_level_values('segment'))
    n_samples = int(len(segment_names)*split_ratio)
    part1_names = set(random.sample(segment_names, n_samples))
    part2_names = segment_names - part1_names
    return list(sorted(part1_names)), list(sorted(part2_names))


def split_experiment_data(interictal, preictal,
                          training_ratio, do_downsample=True,
                          downsample_ratio=2.0,
                          do_segment_split=True):
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
                                        do_segment_split=do_segment_split)
    return split_dataset(dataset,
                         training_ratio=training_ratio,
                         do_segment_split=do_segment_split)


def merge_interictal_preictal(interictal, preictal,
                              do_downsample=True,
                              downsample_ratio=2.0,
                              do_segment_split=True):
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
    if do_downsample:
        interictal = downsample(interictal, preictal,
                                downsample_ratio,
                                do_segment_split=do_segment_split)
    dataset = pd.concat((interictal, preictal))
    dataset.sortlevel('segment', inplace=True)
    return dataset


def downsample(df1, df2, max_skew=1.0, do_segment_split=True):
    """
    Returns a downsampled version of *df1* so that it contains at most
    a ratio of *max_skew* samples of df2.
    Args:
        *df1*: The dataframe which should be downsampled.
        *df2*: The dataframe of the smaller class, its length will be the target
               for the downsampling.
        *max_skew*: The size difference ratio between the downsampled *df1*
                    and *df2*
        *do_segment_split*: Whether the downsampling should be done per segment.
    Returns:
        A slice of df1 containing len(df2)*max_skew number of samples.
    """
    if do_segment_split:
        df1_segments = set(df1.index.get_level_values('segment'))
        df2_segments = set(df2.index.get_level_values('segment'))

        n_samples = int(len(df2_segments)*max_skew)
        n_samples = min(n_samples, len(df1_segments))

        sample_segments = random.sample(set(df1_segments), n_samples)
        return df1.loc[sample_segments]
    else:
        n_samples = int(len(df2)*max_skew)
        n_samples = min(n_samples, len(df1))

        sample_indices = random.sample(range(len(df1)), n_samples)
        return df1.iloc[sample_indices]


def split_dataset(dataset, training_ratio=.8, do_segment_split=True):
    """
    Splits the dataset into a training and test partition.
    Args:
        *dataset*: A data frame to split.
        *training_ratio*: The ratio of the data to use for the first part.
        *do_segment_split*: If True, the split will be done on whole segments.
    Returns:
        A pair of disjunct data frames, where the first frame contain
        *training_ratio* of all the data.
    """

    if do_segment_split:
        training_segments, test_segments = split_segment_names(dataset,
                                                               training_ratio)
        return dataset.loc[training_segments], dataset.loc[test_segments]
    else:
        n_samples = int(len(dataset)*training_ratio)
        train_indices = set(random.sample(range(len(dataset)), n_samples))
        test_indices = set(range(len(dataset))) - train_indices

        return dataset.iloc[list(train_indices)], dataset.iloc[list(test_indices)]
