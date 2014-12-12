"""
Module for loading and manipulating  EEG segments.
"""
from __future__ import absolute_import
import os.path
import glob

import scipy.io
import scipy.signal
import scipy.stats
import pandas as pd
import numpy as np

from . import fileutils


def load_segment(segment_path, old_segment_format=True, normalize_signal=False, resample_frequency=None):
    """
    Convienience function for loading segments
    :param segment_path: Path to the segment file to load.
    :param old_segment_format: If True, the old format will be used. If False, the format backed by a pandas
                               dataframe will be used.
    :param normalize_signal: If True, the signal will be normalized using the subject median and median absolute
                             deviation.
    :param resample_frequency: If this is set to a number, the signal will be resampled to that frequency.
    :return: A Segment or DFSegment object with the data from the segment in *segment_path*.
    """
    if normalize_signal:
        return load_and_standardize(segment_path, old_segment_format=old_segment_format)
    else:
        if old_segment_format:
            segment = Segment(segment_path)
        else:
            segment = DFSegment.from_mat_file(segment_path)
        if resample_frequency is not None:
            segment.resample_frequency(resample_frequency, inplace=True)
        return segment


def load_and_standardize(mat_filename, stats_glob='../../data/segment_statistics/*.csv',
                         center_name='median', scale_name='mad', old_segment_format=True,
                         k=10):
    """
    Loads the segment given by *mat_name* and returns a standardized version. The values for standardization (scaling
    factor and center values) should be in a segment statistics file in *stats_folder* and must have been produced
    earlier.
    :param mat_filename: A path to the segment to load. Must contain the name of the subject in the file.
    :param stats_glob: A folder which keeps statistics files produced by the module basic_segment_statistics. The file
                        to use will be inferred from the subject name in mat_filename, and a stats_file with the same
                        subject name in it should exist in stats_folder.
    :param center_name: The name of the metric to use as a centering vector (one value for each channel).
    Must correspond to a metric present in stats file.
    :param scale_name: The name of the metric to use as a scaling vector (one value for each channel).
    Must correspond to a metric present in stats file.
    :param old_segment_format: If True, use the old segment format.
    :return: A segment object scaled, centered and trimmed using the values loaded from a file in *stats_folder* whose
    name contains the same subject as mat_filename
    """
    from ..features import basic_segment_statistics

    subject = fileutils.get_subject(mat_filename)
    stats_files = [filename for filename
                   in glob.glob(stats_glob)
                   if subject == fileutils.get_subject(filename)]
    if len(stats_files) != 1:
        raise ValueError("Can't determine which stats file to use"
                         "with the glob {} and the subject {}".format(stats_glob, subject))
    stats = basic_segment_statistics.read_stats(stats_files[0])
    center = basic_segment_statistics.get_subject_metric(stats, center_name)
    scale = basic_segment_statistics.get_subject_metric(stats, scale_name)

    if old_segment_format:
        segment = Segment(mat_filename)
    else:
        segment = DFSegment.from_mat_file(mat_filename)

    segment.center(center)
    segment.winsorize(scale, k=k)  # We have to winsorize before scaling
    segment.scale(scale)
    return segment


class Segment:
    """Wrapper class for EEG segments backed by a multidimensional numpy array."""

    def __init__(self, mat_filename):
        """Creates a new segment object from the file named *mat_filename*"""
        # First extract the variable name of the struct, there should be exactly one struct in the file
        try:
            [(struct_name, shape, dtype)] = scipy.io.whosmat(mat_filename)
            if dtype != 'struct':
                raise ValueError("File {} does not contain a struct".format(mat_filename))

            self.filename = os.path.basename(mat_filename)
            self.dirname = os.path.dirname(os.path.abspath(mat_filename))
            self.name = struct_name

            # The matlab struct contains the variable mappings, we're only interested in the variable *self.name*
            self.mat_struct = scipy.io.loadmat(mat_filename, struct_as_record=False, squeeze_me=True)[self.name]
            self.mat_struct.data = self.mat_struct.data.astype('float64')

        except ValueError as exception:
            print("Error when loading {}".format(mat_filename))
            raise exception

    def get_name(self):
        return self.name

    def get_filename(self):
        return self.filename

    def get_dirname(self):
        return self.dirname

    def get_channels(self):
        return self.mat_struct.channels

    def get_n_samples(self):
        """Returns the number of samples in this segment"""
        return self.mat_struct.data.shape[1]

    def get_duration(self):
        """Returns the length of this segment in seconds"""
        return self.get_n_samples() / self.get_sampling_frequency()

    def get_channel_data(self, channel, start_time=None, end_time=None):
        """Returns all data of the given channel as a numpy array.
        *channel* can be either the name of the channel or the index of the channel.
        If *start_time* or *end_time* is given in seconds, only the data corresponding to that window will be returned.
        If *start_time* is after the end of the segment, nothing is returned."""
        if isinstance(channel, int):
            index = channel
        else:
            index = list(self.get_channels()).index(str(channel))

        if start_time is not None or end_time is not None:
            if start_time is None:
                start_index = 0
            else:
                start_index = int(np.floor(start_time * self.get_sampling_frequency()))

            if end_time is None:
                end_index = self.mat_struct.data.shape[1]
            else:
                end_index = int(np.ceil(end_time * self.get_sampling_frequency()))

            return self.get_data()[index][start_index:end_index]
        else:
            return self.get_data()[index]

    def get_data(self):
        return self.mat_struct.data

    def get_length_sec(self):
        return self.mat_struct.data_length_sec

    def get_sampling_frequency(self):
        return self.mat_struct.sampling_frequency

    def get_sequence(self):
        return self.mat_struct.sequence

    def resample_frequency(self, new_frequency, method='resample', inplace=True, **method_kwargs):
        """
        Resample the signal to a new frequency.

        :param new_frequency: The frequency to downsample to. For *method* = 'decimate', it should be lower than
        the current frequency.
        :param method: The method to use for resampling, should be either of 'resample' or 'decimate', corresponding to
        *scipy.signal.resample* and *scipy.signal.decimate* respectively.
        :param inplace: Whether the resampled segment values should replace the current one. If False, a new DFSegment
        object will be returned.
        :param method_kwargs: Key-word arguments to pass to the resampling method, see *scipy.signal.resample* and
        *scipy.signal.decimate* for details.
        :return: A DFSegment with the new frequency. If inplace=True, the calling object will be returned, otherwise a
        newly constructed segment is returned.
        """

        if not inplace:
            raise ValueError("Resample on Segment only supports inplace")
        data = self.mat_struct.data
        if method == 'resample':
            print("Using scipy.signal.resample")
            # Use scipy.signal.resample
            n_samples = int(self.get_n_samples() * new_frequency / self.mat_struct.sampling_frequency)
            resampled_signal = scipy.signal.resample(data, n_samples, axis=1, **method_kwargs)
        elif method == 'decimate':
            print("Using scipy.signal.decimate")
            # Use scipy.signal.decimate
            decimation_factor = int(round(self.mat_struct.sampling_frequency / new_frequency))
            # Since the decimate factor has to be an int, the actual new frequency isn't necessarily the in-argument
            adjusted_new_frequency = self.mat_struct.sampling_frequency / decimation_factor
            if adjusted_new_frequency != new_frequency:
                print("Because of rounding, the actual new frequency is {}".format(adjusted_new_frequency))
            new_frequency = adjusted_new_frequency
            resampled_signal = scipy.signal.decimate(data,
                                                     decimation_factor,
                                                     axis=1,
                                                     **method_kwargs)
        else:
            raise ValueError("Resampling method {} is unknown.".format(method))
        self.mat_struct.data = resampled_signal
        self.mat_struct.sampling_frequency = new_frequency

    def winsorize(self, scale, k=5):
        """Makes any sample which is more than *k* scale units (std or mad) away from the centers, be exactly *k* scale
        units from the center.
        :param scale: A (n_channels, 1) NDArray-like with scale-estimates (like standard deviation) for the channels.
        :param k: The maximum number of standard deviations a value is allowed to have to not be clipped
        :return: None. The winsorizing is done inplace
        """

        limits = k * scale
        outliers = np.abs(self.mat_struct.data) > limits
        limited = np.sign(self.mat_struct.data) * limits
        assert isinstance(limited, np.ndarray)
        self.mat_struct.data[outliers] = limited[outliers]

    def center(self, center):
        """
        Centers the data at the given centers.
        :param center: A NDArray-like of shape (n_channels, 1) with a center for each of the channels.
        :return: None, the centering is done inplace
        """
        self.mat_struct.data = self.mat_struct.data - center

    def scale(self, scale):
        """
        Center and scale the signal.
        :param scale: A (n_channels, 1) NDArray-like with scale-estimates (like standard deviation or median absolute
                      deviations) for the channels.
        :return: None. The scaling is done in-place.
        """
        self.mat_struct.data = self.mat_struct.data / scale

    def mean(self):
        return np.mean(self.mat_struct.data, axis=1)[:, np.newaxis]

    def median(self):
        return np.median(self.mat_struct.data, axis=1)[:, np.newaxis]

    def mad(self, median):
        """Return the median absolute deviation for this segment only,
        scaled by phi-1(3/4) to approximate the standard deviation."""
        c = scipy.stats.norm.ppf(3 / 4)
        subject_median = self.median()
        median_residuals = self.mat_struct.data - subject_median  # deviation between median and data
        mad = np.median(np.abs(median_residuals), axis=1)[:, np.newaxis]
        return mad / c


class DFSegment(object):
    def __init__(self, sampling_frequency, dataframe, do_downsample=False, downsample_frequency=200):
        self.sampling_frequency = sampling_frequency
        self.dataframe = dataframe
        if do_downsample:
            self.resample_frequency(downsample_frequency, inplace=True)

    def get_channels(self):
        return self.dataframe.columns

    def get_n_samples(self):
        """Returns the number of samples in this segment"""
        return len(self.dataframe)

    def get_duration(self):
        """Returns the length of this segment in seconds"""
        return self.get_n_samples() / self.get_sampling_frequency()

    def get_channel_data(self, channel, start_time=None, end_time=None):
        """Returns all data of the given channel as a numpy array.
        *channel* can be either the name of the channel or the index of the channel.
        If *start_time* or *end_time* is given in seconds, only the data corresponding to that window will be returned.
        If *start_time* is after the end of the segment, nothing is returned."""
        if isinstance(channel, int):
            channel_index = self.get_channels()[channel]
        else:
            channel_index = channel

        if start_time is None and end_time is None:
            # Return the whole channel
            return self.dataframe.loc[:, channel_index]
        else:
            if start_time is None:
                start_index = 0
            else:
                # Calculate which index is the first
                start_index = int(np.floor(start_time * self.get_sampling_frequency()))

            if end_time is None:
                end_index = self.get_n_samples()
            else:
                end_index = int(np.ceil(end_time * self.get_sampling_frequency()))
            return self.dataframe.ix[start_index:end_index, channel_index]

    def get_data(self, start_time=None, end_time=None):
        if start_time is None and end_time is None:
            return self.dataframe.transpose()
        else:
            if start_time is None:
                start_index = 0
            else:
                # Calculate which index is the first
                start_index = int(np.floor(start_time * self.get_sampling_frequency()))

            if end_time is None:
                end_index = self.get_n_samples()
            else:
                end_index = int(np.ceil(end_time * self.get_sampling_frequency()))
            return self.dataframe.iloc[start_index:end_index].transpose()

    def get_length_sec(self):
        return self.get_duration()

    def get_sampling_frequency(self):
        return self.sampling_frequency

    def get_dataframe(self):
        return self.dataframe

    def resample_frequency(self, new_frequency, method='resample', inplace=False, **method_kwargs):
        # TODO Code duplication with the other resample_frequency function here
        """
        Resample the signal to a new frequency.

        :param new_frequency: The frequency to downsample to. For *method* = 'decimate', it should be lower than
        the current frequency.
        :param method: The method to use for resampling, should be either of 'resample' or 'decimate', corresponding to
        *scipy.signal.resample* and *scipy.signal.decimate* respectively.
        :param inplace: Whether the resampled segment values should replace the current one. If False, a new DFSegment
        object will be returned.
        :param method_kwargs: Key-word arguments to pass to the resampling method, see *scipy.signal.resample* and
        *scipy.signal.decimate* for details.
        :return: A DFSegment with the new frequency. If inplace=True, the calling object will be returned, otherwise a
        newly constructed segment is returned.
        """

        if method == 'resample':
            print("Using scipy.signal.resample")
            # Use scipy.signal.resample
            n_samples = int(len(self.dataframe) * new_frequency / self.sampling_frequency)
            resampled_signal = scipy.signal.resample(self.dataframe, n_samples, **method_kwargs)
        elif method == 'decimate':
            print("Using scipy.signal.decimate")
            # Use scipy.signal.decimate
            decimation_factor = int(self.sampling_frequency / new_frequency)
            # Since the decimate factor has to be an int, the actual new frequency isn't necessarily the in-argument
            adjusted_new_frequency = self.sampling_frequency / decimation_factor
            if adjusted_new_frequency != new_frequency:
                print("Because of rounding, the actual new frequency is {}".format(adjusted_new_frequency))
            new_frequency = adjusted_new_frequency
            resampled_signal = scipy.signal.decimate(self.dataframe,
                                                     decimation_factor,
                                                     axis=0,
                                                     **method_kwargs)
        else:
            raise ValueError("Resampling method {} is unknown.".format(method))

        # We should probably reconstruct the index
        # index = pd.MultiIndex.from_product([[filename], [sequence],
        # np.arange(mat_struct.data.shape[1])], names=['filename', 'sequence', 'index'])
        resampled_dataframe = pd.DataFrame(data=resampled_signal, columns=self.dataframe.columns)
        if inplace:
            self.sampling_frequency = new_frequency
            self.dataframe = resampled_dataframe
            return self
        else:
            return DFSegment(new_frequency, resampled_dataframe)

    def get_windowed(self, window_length, start_time=None, end_time=None):
        """Returns an iterator with windows of this segment. If *segment_start* or *segment_end* is supplied,
        only windows within this interval will be returned."""
        if start_time is None:
            start_index = 0
        else:
            start_index = int(np.floor(start_time * self.get_sampling_frequency()))

        if end_time is None:
            end_index = self.get_n_samples()
        else:
            end_index = int(np.ceil(end_time * self.get_sampling_frequency()))

        window_sample_length = int(np.floor(window_length * self.get_sampling_frequency()))

        for window_start in np.arange(start_index, end_index - window_sample_length, window_sample_length):
            yield self.dataframe.iloc[window_start: window_start + window_sample_length]

    @classmethod
    def from_mat_file(cls, mat_filename):
        try:
            [(struct_name, shape, dtype)] = scipy.io.whosmat(mat_filename)
            if dtype != 'struct':
                raise ValueError("File {} does not contain a struct".format(mat_filename))

            filename = os.path.basename(mat_filename)

            # The matlab struct contains the variable mappings, we're only interested in the variable *name*
            mat_struct = scipy.io.loadmat(mat_filename, struct_as_record=False, squeeze_me=True)[struct_name]
            sampling_frequency = mat_struct.sampling_frequency
            try:
                sequence = mat_struct.sequence
            except AttributeError:
                sequence = 0

            index = pd.MultiIndex.from_product([[filename], [sequence], np.arange(mat_struct.data.shape[1])],
                                               names=['filename', 'sequence', 'index'])
            ## Most operations we do on dataframes are optimized for float64
            dataframe = pd.DataFrame(data=mat_struct.data.transpose().astype('float64'), columns=mat_struct.channels,
                                     index=index)
            return cls(sampling_frequency, dataframe)

        except ValueError as exception:
            print("Error when loading {}".format(mat_filename))
            raise exception

    @classmethod
    def from_mat_files(cls, file_names):
        """Loads all files in the sequence *file_names* and concatenates them into a single segment"""
        return concat([cls.from_mat_file(file_name) for file_name in file_names])


def concat(segments):
    """Concatenates DFSegments."""
    new_df = pd.concat([s.dataframe for s in segments])
    sampling_frequency = segments[0].get_sampling_frequency()
    new_df.sortlevel(inplace=True)
    return DFSegment(sampling_frequency, new_df)


def test_segment_classes():
    s_new = DFSegment.from_mat_file('../../data/Dog_1/Dog_1_preictal_segment_0001.mat')
    s_old = Segment('../../data/Dog_1/Dog_1_preictal_segment_0001.mat')
    print('Matching durations: ', s_old.get_duration() == s_new.get_duration())
    for start in np.arange(0, s_new.get_duration(), 9.3):
        for channel in s_new.get_channels():
            c_new = s_new.get_channel_data(channel, start, start + 9.3)
            c_old = s_old.get_channel_data(channel, start, start + 9.3)
            print('Length of new: {}, length of old: {}'.format(len(c_new), len(c_old)))
            print('Data matching for channel {}: {}'.format(channel, all(c_new == c_old)))


def example_preictal():
    return Segment('../../data/Dog_1/Dog_1_preictal_segment_0001.mat')


def example_interictal():
    return Segment('../../data/Dog_1/Dog_1_interictal_segment_0001.mat')


if __name__ == '__main__':
    example = load_and_standardize('../../data/Dog_1/Dog_1_preictal_segment_0001.mat')
