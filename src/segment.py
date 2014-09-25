__author__ = 'erik'

import scipy.io
import os.path
import pandas
import numpy as np

class Segment:
    def __init__(self, mat_filename):
        """Creates a new segment object from the file named *mat_filename*"""
        #First extract the variable name of the struct, there should be exactly one struct in the file
        try:
            [(struct_name, shape, dtype)] = scipy.io.whosmat(mat_filename)
            if dtype != 'struct':
                raise ValueError("File {} does not contain a struct".format(mat_filename))

            self.filename = os.path.basename(mat_filename)
            self.dirname = os.path.dirname(os.path.abspath(mat_filename))
            self.name = struct_name

            #The matlab struct contains the variable mappings, we're only interested in the variable *self.name*
            self.mat_struct = scipy.io.loadmat(mat_filename, struct_as_record=False, squeeze_me=True)[self.name]
            self.mat_struct.data = self.mat_struct.data.astype('float32')


        except ValueError as e:
            print("Error when loading {}".format(mat_filename))
            raise e
        self.dataframe = pandas.DataFrame(self.mat_struct.data.transpose().astype('float32'), columns=self.mat_struct.channels)

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
        return self.get_n_samples() * self.get_sampling_frequency()

    def get_channel_data(self, channel, start_time=None, end_time=None):
        """Returns all data of the given channel as a numpy array.
        *channel* can be either the name of the channel or the index of the channel.
        If *start_time* or *end_time* is given in seconds, only the data corresponding to that window will be returned.
        If *start_time* is after the end of the segment, nothing is returned."""
        if isinstance(channel, str):
            index = list(self.get_channels()).index(channel)
        else:
            index = channel

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

    def get_dataframe(self):
        return self.dataframe




def example_preictal():
    return Segment('../data/Dog_1/Dog_1_preictal_segment_0001.mat')


def example_interictal():
    return Segment('../data/Dog_1/Dog_1_interictal_segment_0001.mat')

