__author__ = 'erik'

import scipy.io
import os.path
import pandas

class Segment:
    def __init__(self, mat_filename):
        """Creates a new segment object from the file named *mat_filename*"""
        #First extract the variable name of the struct, there should be exactly one struct in the file
        [(struct_name, shape, dtype)] = scipy.io.whosmat(mat_filename)
        if dtype != 'struct':
            raise ValueError("File {} does not contain a struct".format(mat_filename))

        self.filename = os.path.basename(mat_filename)
        self.dirname = os.path.dirname(os.path.abspath(mat_filename))
        self.name = struct_name

        #The matlab struct contains the variable mappings, we're only interested in the variable *self.name*
        self.mat_struct = scipy.io.loadmat(mat_filename, struct_as_record=False, squeeze_me=True)[self.name]
        self.dataframe = pandas.DataFrame(self.mat_struct.data.transpose(), columns=self.mat_struct.channels)

    def get_name(self):
        return self.name

    def get_filename(self):
        return self.filename

    def get_dirname(self):
        return self.dirname

    def get_channels(self):
        return self.mat_struct.channels

    def get_data(self):
        return self.mat_struct.data

    def get_length_sec(self):
        return self.mat_struct.data_length_sec

    def get_sampling_frequency(self):
        return self.mat_struct.sampling_frequency

    def get_sequence(self):
        return self.mat_struct.sequence




def test_preictal():
    return Segment('../data/Dog_1/Dog_1_preictal_segment_0001.mat')


def test_interictal():
    return Segment('../data/Dog_1/Dog_1_interictal_segment_0001.mat')

