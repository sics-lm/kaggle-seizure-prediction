"""Python module for manipulating datasets."""
import random

import pandas as pd
import numpy as np
import sklearn
from sklearn import cross_validation


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
        interictal = downsample(interictal, len(preictal) * downsample_ratio,
                                do_segment_split=do_segment_split)
    dataset = pd.concat((interictal, preictal))
    dataset.sortlevel('segment', inplace=True)
    return dataset


def downsample(df1, n_samples, do_segment_split=True):
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
    if do_segment_split:
        df1_segments = set(df1.index.get_level_values('segment'))
        samples_per_segment = len(df1)/len(df1_segments)

        n_segment_samples = int(n_samples / samples_per_segment)
        if n_segment_samples < len(df1_segments):
            sample_segments = random.sample(set(df1_segments), n_segment_samples)
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


def split_dataset(dataframe, training_ratio=.8, do_segment_split=True, shuffle=False):
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
        cv = SegmentCrossValidator(dataframe, n_folds=k, shuffle=shuffle)
    else:
        # Don't split by segment, but still do a stratified split
        cv = cross_validation.StratifiedKFold(dataframe['Preictal'], n_folds=k, shuffle=shuffle)

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
    cv = SegmentCrossValidator(dataframe, n_folds = 6)
    for training_fold, test_fold in cv:
        print("Training indice: ", training_fold)
        print("Test indice: ", test_fold)



if __name__ == '__main__':
    test_k_fold_segment_split()
