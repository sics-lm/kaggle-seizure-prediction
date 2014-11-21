from __future__ import absolute_import

import mne
mne.set_log_level(verbose='WARNING')

import sys
from itertools import chain

from . import feature_extractor
from . import wavelets

from .transforms import FFTWithTimeFreqCorrelation as FFT_TF_xcorr
from .transforms import FilteredFFTWithTFCorrelation as Filtered_TF_xcorr

def extract_features_for_segment(
    segment, transformation=None,feature_length_seconds=60, window_size=5, **kwargs):
    """
    Creates a feature dictionary from a Segment object, according to the provided
    transformation function
    Args:
        segment: A Segment object containing the EEG segment from which we want
        to extract the features
        transformation: A class that should implement apply(data), which takes
        an ndarray (n_channels x n_samples) and returns a 1d ndarray of features.
        feature_length_seconds: The number of seconds each frame should consist
        of, should be exactly divisible by window_size.
        window_size: The length of a window in seconds.
    Returns:
        A dict of features, where each keys are the frames indexes in the segment
        and the values are a List of doubles containing all the feature values
        for that frame.
        Ex. For a 10 min segment with feature_length_seconds=60 (sec) we should
        get 10 frames. The length of the lists then depends on the window_size,
        number of channels and number of frequency bands we are examining.
    """

    if transformation is None:
        # transformation = FFT_TF_xcorr(1, 48, 400, 'usf')
        transformation = Filtered_TF_xcorr(
            1, 48, 400, 'usf', segment.get_sampling_frequency())
    # TODO: Assert that the function implements
    # else:

    # Here we define how many windows we will have to concatenate
    # in order to create the features we want
    windows_in_frame = int(feature_length_seconds / window_size)
    total_windows = int(segment.get_duration() / window_size)
    n_channels = len(segment.get_channels())
    iters = int(segment.get_duration() / feature_length_seconds)

    # Create Epochs object according to defined window size
    epochs = wavelets.epochs_from_segment(segment, window_size)

    feature_list = []
    # Create a list of features
    for epoch in epochs:
        feature_list.append(transformation.apply(epoch).tolist())

    feature_dict = {}
    # Slice the features to frames
    for i in range(iters):
        window_features = feature_list[i*windows_in_frame:(i+1)*windows_in_frame]
        feature_dict[i] = list(chain.from_iterable(window_features))


    if len(feature_dict) != iters:
        sys.stderr.write("WARNING: Wrong number of features created, expected"
                         " %d, got %d instead." % (iters, len(feature_dict)))

    return feature_dict


def get_transform(transformation=None, **kwargs):
    if transformation is None:
        return FFT_TF_xcorr(1, 48, 400, 'usf')
    else:
        transformation(**kwargs)


def extract_features(segment_paths,
                     output_dir,
                     workers=1,
                     sample_size=None,
                     old_segment_format=True,
                     resample_frequency=None,
                     normalize_signal=False,
                     only_missing_files=True,
                     feature_length_seconds=60,
                     window_size=5):
    feature_extractor.extract(segment_paths,
                              extract_features_for_segment,
                              ## Arguments for feature_extractor.extract
                              output_dir=output_dir,
                              workers=workers,
                              sample_size=sample_size,
                              old_segment_format=old_segment_format,
                              resample_frequency=resample_frequency,
                              normalize_signal=normalize_signal,
                              only_missing_files=only_missing_files,
                              ## Worker function kwargs:
                              feature_length_seconds=feature_length_seconds,
                              window_size=window_size)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Calculates features according to Mike Hills winning submission.")

    parser.add_argument("segments", help="The files to process. This can either be the path to a matlab file holding the segment or a directory holding such files.", nargs='+', metavar="SEGMENT_FILE")
    parser.add_argument("--csv-directory", help="Directory to write the csv files to, if omitted, the files will be written to the same directory as the segment")
    parser.add_argument("--window-size", help="What length in seconds the epochs should be.", type=float, default=5.0)
    parser.add_argument("--feature-length", help="The length of the feature vectors in seconds, will be produced by concatenating the phase lock values from the windows.", type=float, default=60.0)
    parser.add_argument("--workers", help="The number of worker processes used for calculating the cross-correlations.", type=int, default=1)
    parser.add_argument("--resample-frequency", help="The frequency to resample to,",
                        type=float,
                        dest='resample_frequency')
    parser.add_argument("--normalize-signal",
                        help="Setting this flag will normalize the channels based on the subject median and MAD",
                        default=False,
                        action='store_true',
                        dest='normalize_signal')
    args = parser.parse_args()

    extract_features(args.segments,
                     args.csv_directory,
                     workers=args.workers,
                     resample_frequency=args.resample_frequency,
                     normalize_signal=args.normalize_signal,
                     feature_length_seconds=args.feature_length,
                     window_size=args.window_size)