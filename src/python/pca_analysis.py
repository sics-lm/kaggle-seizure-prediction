import os.path
import pickle
import datetime
import pandas as pd

from wavelet_classification import load_data_frames, random_split

import correlation_convertion
import dataset

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def run_pca_analysis(feature_folder, do_downsample=True, n_samples=100):
    interictal, preictal, test_data = load_data_frames(feature_folder)

    fig,pca = mould_data(interictal, preictal, test_data, do_downsample=do_downsample, n_samples=n_samples)

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    pca_name = "pca_analysis_{}".format(timestamp)
    fig_path = os.path.join(feature_folder, pca_name+".pdf")
    pca_obj_path = os.path.join(feature_folder, pca_name + '.pickle')

    fig.savefig(fig_path)
    with open(pca_obj_path, 'wb') as fp:
        pickle.dump(pca, fp)

    return fig,pca


def run_xcorr_pca_analysis(feature_folder,
                           frame_length=1,
                           do_downsample=True,
                           n_samples=100):
    interictal, preictal, test_data = correlation_convertion.load_data_frames(feature_folder, frame_length=frame_length)
    fig,pca = mould_data(interictal, preictal, test_data, do_downsample=do_downsample, n_samples=n_samples)

    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    pca_name = "pca_analysis_frame_length_{}_{}".format(frame_length, timestamp)
    fig_path = os.path.join(feature_folder, pca_name+".pdf")
    pca_obj_path = os.path.join(feature_folder, pca_name + '.pickle')

    fig.savefig(fig_path)
    with open(pca_obj_path, 'wb') as fp:
        pickle.dump(pca, fp)

    return fig,pca


def mould_data(interictal, preictal, test_data, do_downsample=True, n_samples=100):
    if do_downsample:
        interictal = dataset.downsample(interictal, n_samples, do_segment_split=False)
        preictal = dataset.downsample(preictal, n_samples, do_segment_split=False)
        test_data = dataset.downsample(test_data, n_samples, do_segment_split=False)
    return pca_transform(interictal, preictal, test_data)


def pca_transform(interictal, preictal, test_data, label=None):
    concat_frames = [interictal.drop('Preictal', axis=1), preictal.drop('Preictal', axis=1), test_data]
    feature_matrix = pd.concat(concat_frames)

    pca = PCA(n_components=2)
    trans_pca = pca.fit_transform(feature_matrix)

    interictal_start = 0
    interictal_end = len(interictal)

    preictal_start = interictal_end
    preictal_end = preictal_start + len(preictal)

    test_data_start = preictal_end
    test_data_end = test_data_start + len(test_data)

    fig = plt.figure()
    plt.plot(trans_pca[interictal_start:interictal_end,0],
             trans_pca[interictal_start:interictal_end,1], 'o', markersize=7,
             color='blue', label='Interictal')
    plt.plot(
        trans_pca[preictal_start:preictal_end,0],
        trans_pca[preictal_start:preictal_end,1],
        '^', markersize=7, color='red', alpha=0.5, label='Preictal')
    plt.plot(
        trans_pca[test_data_start:test_data_end,0],
        trans_pca[test_data_start:test_data_end,1],
        'x', markersize=7, color='green', alpha=0.5, label='Test')

    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

    return fig, pca
