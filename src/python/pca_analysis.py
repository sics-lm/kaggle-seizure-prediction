from __future__ import print_function

import sys
import os.path
import pickle
import datetime
import pandas as pd
import numpy as np

from wavelet_classification import load_data_frames, random_split

import correlation_convertion
import dataset

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def has_nan(df):
    return np.count_nonzero(np.isnan(df)) != 0

def run_pca_analysis(feature_folder,
                     do_downsample=True,
                     n_samples=100,
                     do_standardize=False,
                     frame_length=12,
                     sliding_frames=False):
    interictal, preictal, test_data = load_data_frames(feature_folder,
                                                       frame_length=frame_length,
                                                       sliding_frames=sliding_frames)

    if has_nan(interictal) or has_nan(preictal) or has_nan(test_data):
        print("WARNING: NaN values found, quitting!",
              file=sys.stderr)
        sys.exit(1)


    fig, pca = mould_data(interictal, preictal, test_data, do_downsample=do_downsample, n_samples=n_samples)

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    pca_name = "pca_analysis_{}".format(timestamp)
    fig_path = os.path.join(feature_folder, pca_name+".pdf")
    pca_obj_path = os.path.join(feature_folder, pca_name + '.pickle')

    fig.savefig(fig_path)
    with open(pca_obj_path, 'wb') as fp:
        pickle.dump(pca, fp)

    return fig, pca


def run_xcorr_pca_analysis(feature_folder,
                           frame_length=1,
                           do_downsample=True,
                           n_samples=100,
                           do_standardize=False,
                           sliding_frames=False):
    interictal, preictal, test_data = correlation_convertion.load_data_frames(feature_folder,
                                                                              frame_length=frame_length,
                                                                              sliding_frames=sliding_frames)
    fig,pca = mould_data(interictal, preictal, test_data, do_downsample=do_downsample, n_samples=n_samples)

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    pca_name = "pca_analysis_frame_length_{}_{}".format(frame_length, timestamp)
    fig_path = os.path.join(feature_folder, pca_name+".pdf")
    pca_obj_path = os.path.join(feature_folder, pca_name + '.pickle')

    fig.savefig(fig_path)
    with open(pca_obj_path, 'wb') as fp:
        pickle.dump(pca, fp)

    return fig, pca


def mould_data(interictal, preictal, test_data, do_downsample=True, n_samples=100, do_standardize=False):
    if do_downsample:
        interictal = dataset.downsample(interictal, n_samples, do_segment_split=False)
        preictal = dataset.downsample(preictal, n_samples, do_segment_split=False)
        test_data = dataset.downsample(test_data, n_samples, do_segment_split=False)
    return pca_transform(interictal, preictal, test_data, do_standardize=do_standardize)


def pca_transform(interictal, preictal, test_data, label=None, do_standardize=False):
    concat_frames = [interictal.drop('Preictal', axis=1), preictal.drop('Preictal', axis=1), test_data]
    feature_matrix = pd.concat(concat_frames)
    if do_standardize:
        feature_matrix = (feature_matrix - feature_matrix.mean()) / feature_matrix.std()

    pca = PCA(n_components=2)
    trans_pca = pca.fit_transform(feature_matrix)

    interictal_start = 0
    interictal_end = len(interictal)

    preictal_start = interictal_end
    preictal_end = preictal_start + len(preictal)

    test_data_start = preictal_end
    test_data_end = test_data_start + len(test_data)

    fig = plt.figure()
    plt.plot(trans_pca[interictal_start:interictal_end, 0],
             trans_pca[interictal_start:interictal_end, 1], 'o', markersize=7,
             color='blue', label='Interictal')
    plt.plot(
        trans_pca[preictal_start:preictal_end, 0],
        trans_pca[preictal_start:preictal_end, 1],
        '^', markersize=7, color='red', alpha=0.5, label='Preictal')
    plt.plot(
        trans_pca[test_data_start:test_data_end, 0],
        trans_pca[test_data_start:test_data_end, 1],
        'x', markersize=7, color='green', alpha=0.5, label='Test')

    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

    return fig, pca


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="""Script for creating PCA plots of data""")

    parser.add_argument("feature_folder",
                        help="""The folder containing the features""",)

    # parser.add_argument("--rebuild-data",
    #                     action='store_true',
    #                     help="Should the dataframes be re-read from the csv feature files",
    #                     dest='rebuild_data')

    parser.add_argument("--no-downsample",
                        action='store_false',
                        default=True,
                        help="Disable downsampling of the majority class",
                        dest='do_downsample')

    parser.add_argument("--n-samples",
                        default=300,
                        type=int,
                        help="The number of samples to take from each dataset when downsampling",
                        dest='n_samples')

    parser.add_argument("--do-standardize",
                        default=False,
                        action='store_true',
                        help="Should the columns of the data be standardized (zero mean, standard deviation 1)",
                        dest='do_standardize')

    # parser.add_argument("--no-segment-split",
    #                     action='store_false',
    #                     help="Disable splitting data by segment.",
    #                     dest='do_segment_split',
    #                     default=True)

    # parser.add_argument("--processes",
    #                     help="How many processes should be used for parellelized work.",
    #                     dest='processes',
    #                     default=4,
    #                     type=int)

    parser.add_argument("--frame-length",
                        help="The size in windows each frame (feature vector) should be. Only applicable to --feature-type 'xcorr' at the momen",
                        dest='frame_length',
                        default=12,
                        type=int)

    parser.add_argument("--sliding-frames",
                        help="Enable oversampling by using a sliding frame over the windows",
                        dest='sliding_frames',
                        default=False,
                        action='store_true')

    parser.add_argument("--feature-type", help="What kind of features are being processed.", choices=['wavelets', 'xcorr'],
                        default='wavelets')

    args = parser.parse_args()
    if args.feature_type == 'wavelets':
        run_pca_analysis(args.feature_folder, do_downsample=args.do_downsample,
                         n_samples=args.n_samples, do_standardize=args.do_standardize,
                         sliding_frames=args.sliding_frames,
                         frame_length=args.frame_length)
    elif args.feature_type == 'xcorr':
        run_xcorr_pca_analysis(feature_folder=args.feature_folder,
                               frame_length=args.frame_length,
                               do_downsample=args.do_downsample,
                               n_samples=args.n_samples,
                               do_standardize=args.do_standardize,
                               sliding_frames=args.sliding_frames)
