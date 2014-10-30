""" Python module for doing feature extraction in parallel over segments """
import os.path
import csv
import multiprocessing
import random

import fileutils
from segment import Segment, DFSegment



def extract(feature_folder,
            extractor_function,
            output_dir=None,
            old_segment_format=True,
            workers=1,
            naming_function=None,
            sample_size=None,
            only_missing_files=False,
            **extractor_kwargs):
    """ Performs feature extraction of the segment files found in *feature_folder*. The features are written to csv files in *output_directory*
    Args:
        extractor_function: A function to extract the segment data. Should accept a segment object as its first argument.
        output_dir: The directory the features will be written to. Will be created if it doesn't exist.
        old_segment_format: Should the segment object be loaded with the old segment format.
        workers: The numbers of processes to use for extracting features in parallel.
        naming_function: A function to use for generating the name of the feature file, should accept the two positional arguments segment_path and output_dir, as well as the keyword arguments accepted by the extractor function. If a naming functio isn't supplied, a name will be generate based on the name of the extractor function.
        sample_size: optionally sample this many samples from the input files.
        only_missing_files: If True, features will only be generated for files which doesn't have feature files in output_dir already. To determine if the files are present, the naming function (either the default or a user-suppled) will be used.

    Returns:
        None. The feature results are written to output_dir.
    """

    segments = list(sorted(filter(lambda x: '.mat' in x, sorted(fileutils.expand_paths(feature_folder)))))

    if only_missing_files:
        processed_features = set([os.path.join(output_dir, f) for f in os.listdir(output_dir)])
        unprocessed_segments = []
        for segment in segments:
            if naming_function:
                segment_feature = naming_function(segment, output_dir, **extractor_kwargs)
            else:
                segment_feature = default_naming_function(segment_path, output_dir, extractor_function)
            if segment_feature not in processed_features:
                unprocessed_segments.append(segment)

        segments = unprocessed_segments

    if sample_size is not None and sample_size < len(files):
        segments = random.sample(segments, sample_size)

    if workers > 1:
        pool = multiprocessing.Pool(workers)
        try:
            for segment in segments:
                pool.apply_async(worker_function, (segment, extractor_function,  output_dir, old_segment_format, extractor_kwargs, naming_function))
        finally:
            pool.close()
            pool.join()

    else:
        for segment in segments:
            worker_function(segment, extractor_function, output_dir, old_segment_format, extractor_kwargs, naming_function)


def worker_function(segment_path, extractor_function, output_dir, old_segment_format, extractor_kwargs, naming_function=None):
    """Worker function for the feature extractor. Reads the segment
    from *segment_path* and runs uses it as the first argument to
    *extractor_function*"""

    if output_dir is None:
        output_dir = os.path.dirname(segment_path)

    if old_segment_format:
        segment = Segment(segment_path)
    else:
        segment = DFSegment.from_mat_file(segment_path)
    features = extractor_function(segment, **extractor_kwargs)

    if naming_function is None:
        csv_file = default_naming_function(segment_path, output_dir, extractor_function)
    else:
        csv_file = naming_function(segment_path, output_dir, **extractor_kwargs)

    if not os.path.exists(os.path.dirname(csv_file)):
        os.makedirs(os.path.dirname(csv_file))

    with open(csv_file, 'w') as fp:
        if isinstance(features, dict):
            csv_writer = csv.writer(fp)
            for index, feature in sorted(features.items()):
                csv_writer.writerow(feature)
        elif isinstance(features, list):
            csv_writer = csv.DictWriter(fp, fieldnames=features[0].keys(), delimiter='\t')
            csv_writer.writeheader()
            csv_writer.writerows(features)


def default_naming_function(segment_path, output_dir, extractor_function):
    basename, ext = os.path.splitext(os.path.basename(segment_path))
    return  os.path.join(output_dir, "{}_{}.csv".format(basename, extractor_function.__name__))


def test_extractor(segment):
    return { 'channels' : segment.get_channels() }
