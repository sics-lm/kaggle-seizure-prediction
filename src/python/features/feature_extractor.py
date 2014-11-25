""" Python module for doing feature extraction in parallel over segments """
from __future__ import absolute_import
import os.path
import csv
import multiprocessing
import random

from ..datasets import fileutils
from ..datasets import segment as sg


def extract(feature_folder,
            extractor_function,
            output_dir=None,
            old_segment_format=True,
            normalize_signal=False,
            workers=1,
            naming_function=None,
            sample_size=None,
            only_missing_files=False,
            resample_frequency=None,
            **extractor_kwargs):
    """
    Performs feature extraction of the segment files found in *feature_folder*. The features are written to csv
    files in *output_directory*.

    :param feature_folder: The folder in which the feature extraction will be performed. This should be a folder
    containing the .mat data files.
    :param extractor_function: A function to extract the segment data. Should accept a segment object as its first argument.
    :param output_dir: The directory the features will be written to. Will be created if it doesn't exist.
    :param old_segment_format: Should the segment object be loaded with the old segment format.
    :param normalize_signal:
    :param workers: The numbers of processes to use for extracting features in parallel.
    :param naming_function: A function to use for generating the name of the feature file, should accept the two
    positional arguments segment_path and output_dir, as well as the keyword arguments accepted by the extractor function. If a
    naming function isn't supplied, a name will be generate based on the name of the extractor function.
    :param sample_size: optionally sample this many samples from the input files.
    :param only_missing_files: If True, features will only be generated for files which are missing. Useful if you
    started a feature extraction job but it failed before performing the extraction on all files. To determine if the
    files are present, the naming function will be used.
    :param resample_frequency: If this is not None, the segments will be resampled to this frequency.
    :param extractor_kwargs: Keyword arguments for the extractor function
    :return: None. The feature csv files are created by this function.
    """

    segments = [segment_path
                for segment_path
                in sorted(fileutils.expand_paths(feature_folder))
                if 'mat' in segment_path]

    if only_missing_files:
        processed_features = set()
        for dirpath, dirnames, filenames in os.walk(output_dir):
            processed_features.update([os.path.join(dirpath, filename) for filename in filenames if '.csv' in filename])

        unprocessed_segments = []
        for segment in segments:
            if naming_function:
                segment_feature = naming_function(segment, output_dir, **extractor_kwargs)
            else:
                segment_feature = default_naming_function(segment, output_dir, extractor_function)
            if segment_feature not in processed_features:
                unprocessed_segments.append(segment)

        segments = unprocessed_segments

    if sample_size is not None and sample_size < len(segments):
        segments = random.sample(segments, sample_size)

    if workers > 1:
        pool = multiprocessing.Pool(workers)
        try:
            for segment in segments:
                pool.apply_async(worker_function,
                                 kwds=dict(segment_path=segment,
                                           extractor_function=extractor_function,
                                           output_dir=output_dir,
                                           old_segment_format=old_segment_format,
                                           normalize_signal=normalize_signal,
                                           extractor_kwargs=extractor_kwargs,
                                           naming_function=naming_function,
                                           resample_frequency=resample_frequency))
        finally:
            pool.close()
            pool.join()

    else:
        for segment in segments:
            worker_function(segment_path=segment,
                            extractor_function=extractor_function,
                            output_dir=output_dir,
                            old_segment_format=old_segment_format,
                            normalize_signal=normalize_signal,
                            extractor_kwargs=extractor_kwargs,
                            naming_function=naming_function,
                            resample_frequency=resample_frequency)


def worker_function(segment_path, extractor_function, output_dir,
                    old_segment_format=False, normalize_signal=False,
                    extractor_kwargs=None,
                    naming_function=None,
                    resample_frequency=None):
    """
    Worker function for the feature extractor. Reads the segment from *segment_path* and runs uses it as the first
    argument to *extractor_function*.

    :param segment_path: A path to the segment file to work on.
    :param extractor_function: A function which accepts a segment object as its first positional argument.
    :param output_dir: The directory where the resulting features will be written to.
    :param old_segment_format: Toggles which class should be used to represent the segment. If True, a segment.Segment
    object is used, otherwise a segment.DFSegment object is used.
    :param normalize_signal: Whether to normalize the signal before performing the feature extraction
    :param extractor_kwargs: A dictionary of keyword arguments which will be sent to the extractor function as well as
    the naming function.
    :param naming_function: A function to use for generating the file name for this feature. It should accept a segment
    path and output dir as its first arguments. The extractor_kwargs dictionary will also be supplied as key-word
    arguments.
    :param resample_frequency: If this is not None, the segments will be resampled to this frequency.
    :return: None. The features will be written to the file generated by *naming_function*, or *default_naming_function*.
    """
    if extractor_kwargs is None:
        extractor_kwargs = dict()

    if output_dir is None:
        output_dir = os.path.dirname(segment_path)

    segment = sg.load_segment(segment_path,
                           old_segment_format=old_segment_format,
                           normalize_signal=normalize_signal,
                           resample_frequency=resample_frequency)

    features = extractor_function(segment, **extractor_kwargs)
    write_features(features, segment_path, extractor_function, output_dir, extractor_kwargs, naming_function)
    print("Segment {} completed".format(segment_path))


def write_features(features, segment_path, extractor_function, output_dir, extractor_kwargs, naming_function=None):
    """
    Creates the csv output files for the feature extraction

    :param features: A dict containing the extracted features. Each item in the dict corresponds to one frame in the
    extraction
    :param segment_path: A path to the segment file for which the features were extracted.
    :param extractor_function: A function which accepts a segment object as its first positional argument.
    :param output_dir: The directory where the resulting features will be written to.
    :param extractor_kwargs: Keyword arguments for the extractor function
    :param naming_function: A function to use for generating the name of the feature file.
    :return: None. Creates the output csv files
    """
    if naming_function is None:
        csv_file_path = default_naming_function(segment_path, output_dir, extractor_function)
    else:
        csv_file_path = naming_function(segment_path, output_dir, **extractor_kwargs)

    if not os.path.exists(os.path.dirname(csv_file_path)):
        os.makedirs(os.path.dirname(csv_file_path))

    with open(csv_file_path, 'w') as csv_file:
        if isinstance(features, dict):
            csv_writer = csv.writer(csv_file)
            for index, feature in sorted(features.items()):
                csv_writer.writerow(feature)
        elif isinstance(features, list):
            csv_writer = csv.DictWriter(csv_file, fieldnames=features[0].keys(), delimiter='\t')
            csv_writer.writeheader()
            csv_writer.writerows(features)


def default_naming_function(segment_path, output_dir, extractor_function):
    """
    Creates the default names for the extracted feature csv files
    :param segment_path: The path to the segment being extracted
    :param output_dir: The directory where the resulting features will be written to.
    :param extractor_function: A function which accepts a segment object as its first positional argument.
    :return: A String containing the name of the feature file to be extracted
    """
    if fileutils.get_subject(output_dir) is None:
        subject = fileutils.get_subject(segment_path)
        output_dir = os.path.join(output_dir, subject)
    basename, ext = os.path.splitext(os.path.basename(segment_path))
    return os.path.join(output_dir, "{}_{}.csv".format(basename, extractor_function.__name__))


def test_extractor(segment):
    return {'channels': segment.get_channels()}
