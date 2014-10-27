""" Python module for doing feature extraction in parallel over segments """
import os.path
import csv
import multiprocessing

import fileutils
from segment import Segment


def extract(feature_folder, extractor_function, output_dir=None,
            workers=1, naming_function=None, **extractor_kwargs):
    """ Runs the *extractor_function* on all the segments in
    *feature_folder*. If *output_dir* is specified, the csv files
    produced will be saved to that folder, otherwise they will be
    saved to the same folder as the segment file is in. """

    segments = filter(lambda x: '.mat' in x, sorted(fileutils.expand_paths(feature_folder)))

    if workers > 1:
        pool = multiprocessing.Pool(workers)
        try:
            for segment in segments:
                pool.apply_async(worker_function, (segment, extractor_function, output_dir,
                                                   extractor_kwargs, naming_function))
        finally:
            pool.close()
            pool.join()

    else:
        for segment in segments:
            worker_function(segment, extractor_function, output_dir, extractor_kwargs, naming_function)


def worker_function(segment_path, extractor_function, output_dir, extractor_kwargs, naming_function=None):
    """Worker function for the feature extractor. Reads the segment
    from *segment_path* and runs uses it as the first argument to
    *extractor_function*"""

    if output_dir is None:
        output_dir = os.path.dirname(segment_path)

    segment = Segment(segment_path)
    features = extractor_function(segment, **extractor_kwargs)

    basename, ext = os.path.splitext(os.path.basename(segment_path))
    if naming_function is None:
        csv_file = os.path.join(output_dir, "{}_{}.csv".format(basename, extractor_function.__name__))
    else:
        csv_file = naming_function(segment_path, output_dir, **extractor_kwargs)

    with open(csv_file, 'w') as fp:
        if isinstance(features, dict):
            csv_writer = csv.writer(fp)
            csv_writer.writerows(features.values())
        elif isinstance(features, list):
            csv_writer = csv.DictWriter(fp, fieldnames=features[0].keys(), delimiter='\t')
            csv_writer.writeheader()
            csv_writer.writerows(features)


def test_extractor(segment):
    return { 'channels' : segment.get_channels() }
