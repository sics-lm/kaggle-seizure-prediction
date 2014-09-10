__author__ = 'erik'

import scipy.io
import os.path

class Segment:
    def __init__(self, mat_filename):
        """Creates a new segment object from the file named *mat_filename*"""
        #First extract the variable name of the struct, there should be exactly one struct in the file
        [(struct, shape, dtype)] = scipy.io.whosmat(mat_filename)
        if dtype != 'struct':
            raise ValueError("File {} does not contain a struct".format(mat_filename))

        self.filename = os.path.basename(mat_filename)
        self.dirname = os.path.dirname(os.path.abspath(mat_filename))
        self.name = struct
        self.mat_struct = scipy.io.loadmat(mat_filename)[self.name]

    def get_name(self):
        return self.name

    def get_filename(self):
        return self.filename

    def get_dirname(self):
        return self.dirname
    
    def get_channels(self):
        return [channel[0] for channel in self.mat_struct['channels'][0][0][0]]

    def get_data(self):
        return self.mat_struct['data'][0][0]

    def get_length_sec(self):
        return self.mat_struct['data_length_sec'][0][0][0][0]

    def get_sampling_frequency(self):
        return self.mat_struct['sampling_frequency'][0][0][0][0]

    def get_sequence(self):
        return self.mat_struct['sequence'][0][0][0][0]



def test():
    return Segment('../data/Dog_1/Dog_1_preictal_segment_0001.mat')


