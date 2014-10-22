import pandas as pd
import glob
import os.path
import re
import multiprocessing

channel_pattern = re.compile(r'(?:[a-zA-Z0-9]*_)*(c[0-9]*|[A-Z]*_[0-9]*)$')

def convert_channel_name(name):
    """Pass"""
    match = re.match(channel_pattern, name)
    if match:
        return match.group(1) or match.group(2)
    else:
        return name


def load_and_pivot(filename):
    with open(filename) as fp:
        dataframe = pd.read_csv(fp, sep="\t")
        channel_i = dataframe['channel_i'].map(convert_channel_name)
        channel_j = dataframe['channel_j'].map(convert_channel_name)
        dataframe['channels'] = channel_i.str.cat(channel_j, sep=":")
        return dataframe.pivot('start_sample', 'channels', 'correlation')


def load_correlation_files(feature_folder,
                           class_name,
                           file_pattern="5.0s.csv",
                           rebuild_data=False,
                           processes=1):
    cache_file = os.path.join(feature_folder, '{}_cache.pickle'.format(class_name))

    if rebuild_data or not os.path.exists(cache_file):
        print("Rebuilding {} data".format(class_name))
        full_pattern="*{}*{}".format(class_name, file_pattern)
        glob_pattern=os.path.join(feature_folder, full_pattern)
        files=glob.glob(glob_pattern)
        segment_names = [os.path.basename(filename) for filename in files]
        if processes > 1:
            print("Reading files in parallel")
            pool = multiprocessing.Pool(processes)
            try:
                segment_frames = pool.map(load_and_pivot, files)
            finally:
                pool.close()
        else:
            print("Reading files serially")
            segment_frames = [load_and_pivot(filename) for filename in files]
        complete_frame = pd.concat(segment_frames,
                                   keys=segment_names,
                                   names=('segment', 'start_sample'))
                                   keys=segment_names)

        complete_frame.to_pickle(cache_file)
    else:
        complete_frame = pd.read_pickle(cache_file)
    return complete_frame


def load_data_frames(feature_folder, rebuild_data=False,
                     processes=4, file_pattern="5.0s.csv"):

    preictal = load_correlation_files(feature_folder,
                                      class_name="preictal",
                                      file_pattern=file_pattern,
                                      rebuild_data=rebuild_data,
                                      processes=processes)
    preictal['Class'] = "Preictal"

    interictal = load_correlation_files(feature_folder,
                                        class_name="interictal",
                                        file_pattern=file_pattern,
                                        rebuild_data=rebuild_data,
                                        processes=processes)
    interictal['Class'] = "Interictal"

    test = load_correlation_files(feature_folder,
                                  class_name="test",
                                  file_pattern=file_pattern,
                                  rebuild_data=rebuild_data,
                                  processes=processes)

    return interictal, preictal, test


def get_channel_df(dataframe):
    pass
