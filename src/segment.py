__author__ = 'erik'

import scipy.io

class Segment:
    def __init__(self, mat_filename):
        """Creates a new segment object from the file named *mat_filename*"""
        [(struct, shape, dtype)] = scipy.io.whosmat(mat_filename)
        if dtype != 'struct':
            raise ValueError("File {} does not contain a struct".format(mat_filename))

        self.mat_struct = scipy.io.loadmat(mat_filename)[struct]


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


