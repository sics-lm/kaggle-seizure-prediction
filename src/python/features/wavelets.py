from __future__ import absolute_import
import mne

from ..datasets import segment as sg
from . import feature_extractor

mne.set_log_level(verbose='WARNING')
from mne.time_frequency.tfr import cwt_morlet
import random
import sys
import numpy as np
from itertools import chain


class EpochShim(object):
    """A wrapper for our segments which mimicts the interface of mne.Epoch, for the band_wavelet_synchrony function."""
    def __init__(self, segment, window_size):
        self.segment = segment
        self.window_size = window_size
        #The epoch needs a dictionary attribute with the key 'freq'
        self.info = dict(sfreq = segment.get_sampling_frequency())

    def __iter__(self):
        for window in self.segment.get_windowed(self.window_size):
            yield window.transpose()


def epochs_from_segment(segment, window_size=5.0):
    """
    Creates an MNE Epochs object from a Segment object

    Args:
    segment: The segment object we want to convert
    window_size: The size of the window in seconds

    Returns:
    An mne.Epochs object, where each epoch corresponds to a **window_size**
    part of the segment.
    """

    assert isinstance(segment, sg.Segment) or isinstance(segment, sg.DFSegment)

    ch_names = segment.get_channels().tolist()
    ch_types = ['eeg' for _ in range(len(ch_names))]
    sample_rate = segment.get_sampling_frequency()

    info = mne.create_info(ch_names, sample_rate, ch_types)

    # Slice the data so we can reshape array
    # n_samples = segment.get_n_samples()
    # samples_per_epoch = int(n_samples/sample_rate/window_size)
    # n_epochs = n_samples/samples_per_epoch
    # Reshape data into (n_epochs, n_channels, n_times) format
    # Ensure that the reshape is done correctly, ie. we retain continuity of
    # samples
    # reshaped = sliced_data.reshape(n_epochs, size(ch_names), samples_per_epoch)
    # sliced_data = segment.get_data()[:, :(samples_per_epoch*n_epochs)]

    raw = mne.io.RawArray(segment.get_data(), info)

    random_id = int(random.randrange(sys.maxsize))
    events = make_fixed_length_events(raw, random_id,
                                      window_duration=window_size)
    epochs = mne.Epochs(raw, events, event_id=random_id, tmin=0,
                        tmax=window_size)

    return epochs

def make_fixed_length_events(raw, event_id, window_duration=5.):
    """Make a set of events separated by a fixed duration
    Parameters
    ----------
    raw : instance of Raw
    A raw object to use the data from.
    id : int
    The id to use.
    window_duration: float
    The duration to separate events by.
    Returns
    -------
    new_events : numpy array
    The new events.
    """
    start = raw.time_as_index(0)
    start = start[0] + raw.first_samp
    stop = raw.last_samp + 1
    frequency = raw.info['sfreq']

    if not isinstance(event_id, int):
        raise ValueError('event_id must be an integer')
    total_duration = int(np.floor(raw.n_times / frequency))
    floored_samples_per_window = int(np.floor(frequency * window_duration))
    floored_windows_per_segment = int(np.floor(total_duration /
                                               window_duration))

    stop = floored_windows_per_segment * floored_samples_per_window

    event_samples = np.arange(
        start, stop, np.floor(frequency * window_duration)).astype(int)

    n_events = len(event_samples)
    events = np.c_[event_samples, np.zeros(n_events, dtype=int),
                   event_id * np.ones(n_events, dtype=int)]
    return events


def extract_features_for_segment(segment, feature_length_seconds=60, window_size=5, no_epochs=False):
    """
    Creates an SPLV feature dictionary from a Segment object
    Args:
        segment: A Segment object containing the EEG segment from which we want
        to extract the features
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

    # Here we define how many windows we will have to concatenate
    # in order to create the features we want
    frames = int(feature_length_seconds / window_size)
    total_windows = int(segment.get_duration() / window_size)
    n_channels = len(segment.get_channels())
    iters = int(segment.get_duration() / feature_length_seconds)

    # Extract the features for individual frequency bands and windows
    decomposition_dict = segment_wavelet_synchrony(segment, window_size=window_size, no_epochs=no_epochs)

    feature_dict = {}
    # Combine the individual frequency bands and windows into features
    for index, offset in enumerate(range(0, total_windows, frames)):
        feature_list = []
        for band_name, array_list in sorted(decomposition_dict.items()):
            for i in range(frames):
                try:
                    sync_array = array_list[i + offset]
                    index_upper_1 = np.triu_indices(n_channels, 1)
                    sync_values = sync_array[index_upper_1].tolist()
                    feature_list.append(sync_values)
                # Because of the way the files are we will usually end up with
                # fewer frames than the theoretical value in the final segment,
                # so we need to guard against IndexError
                except IndexError:
                    # sys.stderr.write(("Out of index at index:%d offset:%d"
                    #                  " i:%d\n") % (index, offset, i))
                    # break or pass?
                    pass
        # Flatten the list of lists
        feature_dict[index] = list(chain.from_iterable(feature_list))

    if len(feature_dict) != iters:
        sys.stderr.write("WARNING: Wrong number of features created, expected"
                         " %d, got %d instead." % (iters, len(feature_dict)))

    return feature_dict




def eeg_rhythms():
    """
    Returns a dict of the EEG rhythm bands as described in
    [1] "Comparing SVM and Convolutional Networks for Epileptic Seizure
    Prediction from Intracranial EEG"
    """
    return {"delta" : (1, 4), "theta": (4, 7), "alpha" : (7, 13),
            "low-beta" : (13, 15), "high-beta" : (14, 30),
            "low-gamma" : (30, 45), "high-gamma" : (65, 101)}


def segment_wavelet_synchrony(segment, bands=None, window_size=5.0, no_epochs=False):
    """
    Calculates the wavelet synchrony of a Segment object

    Args:
        segment: A Segment object containing the EEG segment of which we want
        create the wavelet transform of.
        bands: A dict containing {band : (start_freq, stop_freq)} String to
        Tuple2 pairs.
        window_size: The size of the windows.
        no_epochs: If this is True, the EpochShim will be used instead of an mne.Epoch

    Returns:
        A dict containing {band: List[av_sync_array]} String to List of
        (n_channels x n_channels) ndarrays.
        Each band corresponds to a List of ndarrays where each array
        corresponds to the channel-to-channel synchrony within an epoch/window.
    """
    if bands is None:
        bands = eeg_rhythms()

    if no_epochs:
        epochs = EpochShim(segment, window_size)
    else:
        epochs = epochs_from_segment(segment, window_size=window_size)

    decomposition_dict = {}

    for band_name, (start_freq, stop_freq) in bands.items():
        decomposition_dict[band_name] = band_wavelet_synchrony(
            epochs, start_freq, stop_freq)

    return decomposition_dict


def band_wavelet_synchrony(epochs, start_freq, stop_freq):
    """
    Computes the phase-locking synchrony SPLV for a specific frequency band,
    by computing the synchrony over all freqs in the [start_freq, stop_freq)
    band and taking the average.

    Args:
        epochs: The Epochs object for which we compute the wavelet synchrony.
        start_freq: The start of the frequency band
        stop_freq: The end of the frequency band, excluded from the calculation

    Returns:
         A List of (n_channels x n_channels) lower-triangular ndarrays.
         Each item in the list corresponds to the phase synchrony between
         the channels for an epoch/window.
    """
    freqs = range(start_freq, stop_freq)
    tf_decompositions = []
    for epoch in epochs:
        # Calculate the Wavelet transform for all freqs in the range
        tfd = cwt_morlet(epoch, epochs.info['sfreq'],
                         freqs, use_fft=True, n_cycles=2)
        n_channels, n_frequencies, n_samples = tfd.shape

        # Calculate the phase synchrony for all frequencies in the range
        av_phase_sync = np.zeros((n_channels, n_channels),
                                 dtype=np.double)
        for frequency_idx in range(n_frequencies):
            freq_tfd = tfd[:, frequency_idx, :]
            freq_phase_diff = np.zeros((n_channels, n_channels),
                                       dtype=np.double)
            for i, ch_i in enumerate(range(0, n_channels)[:-1]):
                for ch_j in range(0, n_channels)[i+1:]:
                    # Get the wavelet coefficients for each channel
                    ch_i_vals = freq_tfd[ch_i, :]
                    ch_j_vals = freq_tfd[ch_j, :]
                    # Phase difference between two channels is derived
                    # from the angle of their wavelet coefficients
                    angles = ((ch_i_vals * ch_j_vals.conjugate()) /
                              (np.absolute(ch_i_vals) * np.absolute(ch_j_vals)))
                    phase_diff = np.absolute(angles.sum() / n_samples)

                    if (phase_diff > 1.0) or (phase_diff < 0.0):
                        sys.stderr.write(("WARNING: Invalid phase difference: "
                                          "%f\n") % phase_diff)
                    # Gather the values in an lower triangular matrix
                    freq_phase_diff[ch_i, ch_j] = phase_diff

            av_phase_sync += freq_phase_diff

        # The synchrony is averaged over the synchronies in all frequencies
        # in the band
        av_phase_sync /= n_frequencies

        tf_decompositions.append(av_phase_sync)

    return tf_decompositions


def extract_features(segment_paths,
                     output_dir,
                     workers=1,
                     sample_size=None,
                     old_segment_format=True,
                     resample_frequency=None,
                     normalize_signal=False,
                     feature_length_seconds=60,
                     window_size=5,
                     no_epochs=False,
                     only_missing_files=True):
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
                              window_size=window_size,
                              no_epochs=no_epochs)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Calculates the SPLV phase lock between pairwise channels.")

    parser.add_argument("segments", help="The files to process. This can either be the path to a matlab file holding the segment or a directory holding such files.", nargs='+', metavar="SEGMENT_FILE")
    parser.add_argument("--csv-directory", help="Directory to write the csv files to, if omitted, the files will be written to the same directory as the segment")
    parser.add_argument("--window-size", help="What length in seconds the epochs should be.", type=float, default=5.0)
    parser.add_argument("--feature-length", help="The length of the feature vectors in seconds, will be produced by concatenating the phase lock values from the windows.", type=float, default=60.0)
    parser.add_argument("--workers", help="The number of worker processes used for calculating the cross-correlations.", type=int, default=1)
    parser.add_argument("--no-epochs", help="Don't use mne Epochs when generating the windows, just use the raw windows from the segment.",
                        action='store_true',
                        dest='no_epochs',
                        default=False)
    parser.add_argument("--resample-frequency", help="The frequency to resample to,",
                        type=float,
                        dest='resample_frequency')

    args = parser.parse_args()

    extract_features(args.segments,
                     ## Arguments for feature_extractor.extract
                     output_dir=args.csv_directory,
                     workers=args.workers,
                     sample_size=args.sample_size,
                     resample_frequency=args.resample_frequency,
                     ## Worker function kwargs:
                     feature_length_seconds=args.feature_length,
                     window_size=args.window_size,
                     no_epochs=args.no_epochs)
