from wavelet_classification import load_data_frames, random_split
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def run_pca_analysis(feature_folder):
    interictal, preictal, _ = load_data_frames(feature_folder)

    preictal_samples = preictal.shape[0]

    downsampled_interictal, _ = random_split(
        interictal, desired_rows=preictal_samples*2, seed=None)

    interictal_samples = downsampled_interictal.shape[0]

    complete = downsampled_interictal.drop(
        'Preictal', axis=1).append(preictal.drop('Preictal', axis=1))

    pca_transform(complete, interictal_samples, preictal_samples)


def pca_transform(feature_matrix, interictal_samples, preictal_samples):

    pca = PCA(n_components=2)
    trans_pca = pca.fit_transform(feature_matrix)

    plt.plot(trans_pca[0:interictal_samples,0],
             trans_pca[0:interictal_samples,1], 'o', markersize=7,
             color='blue', label='Interictal')
    plt.plot(
        trans_pca[interictal_samples:interictal_samples+preictal_samples,0],
        trans_pca[interictal_samples:interictal_samples+preictal_samples,1],
        '^', markersize=7, color='red', alpha=0.5, label='Preictal')

    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

    plt.show()