# Seizure Detection

This repository contains the submission made by members of the Learning Machines group at the Swedish Institute of Computer Science for the American Epilepsy Society Seizure Prediction Challenge on Kaggle.

http://www.kaggle.com/c/seizure-prediction

This README modelled on https://www.kaggle.com/wiki/ModelSubmissionBestPractices and https://github.com/MichaelHills/seizure-detection

##Hardware / OS platform used

 * Feature extraction on 16 cores 128GB RAM server, classification on Core i7, 16GB RAM PC
 * Server: FreeBSD 9.3, PC: Ubuntu 14.04

##Dependencies

###Required

 * Python 2.7
 * scikit_learn-0.14.1
 * numpy-1.8.1
 * pandas-0.14.0
 * scipy
 * mne


##Train the model and make predictions

Obtain the competition data and place it in the root directory of the project.
```
data/
  Dog_1/
    Dog_1_interictal_segment_0001.mat
    Dog_1_interictal_segment_0002.mat
    ...
    Dog_1_preictal_segment_0001.mat
    Dog_1_preictal_segment_0002.mat
    ...
    Dog_1_test_segment_0001.mat
    Dog_1_test_segment_0002.mat
    ...

  Dog_2/
  ...
```

The directory name of the data should match the value in SETTINGS.json under the key `competition-data-dir`.

Then simply run:
```
./train.py
```

One classifier is trained for each patient, and dumped to the data-cache directory.

```
data-cache/classifier_Dog_1_fft-with-time-freq-corr-1-48-r400-usf-gen1_rf3000mss1Bfrs0.pickle
data-cache/classifier_Dog_2_fft-with-time-freq-corr-1-48-r400-usf-gen1_rf3000mss1Bfrs0.pickle
...
data-cache/classifier_Patient_8_fft-with-time-freq-corr-1-48-r400-usf-gen1_rf3000mss1Bfrs0.pickle
```

Although using these classifiers outside the scope of this project is not very straightforward.

More convenient is to run the predict script.

```
./predict.py
```

This will take at least 2 hours. Feel free to update the classifier's `n_jobs` parameter
in `seizure_detection.py`.

A submission file will be created under the directory specified by the `submission-dir` key
in `SETTINGS.json` (default `submissions/`).

Predictions are made using the test segments found in the competition data directory. They
are iterated over starting from 1 counting upwards until no file is found.

i.e.
```
data/
  Dog_1/
    Dog_1_test_segment_0001.mat
    Dog_1_test_segment_0002.mat
    ...
    Dog_1_test_segment_0502.mat
```

To make predictions on a new dataset, simply replace these test segments with new ones.
The files must numbered sequentially starting from 1 otherwise it will not find all of
the files.

## Run cross-validation

```
./cross_validation.py
```

Cross-validation set is obtained by splitting on entire segments. That way we avoid taking multiple samples from the same segment, since they could be correlated.


## SETTINGS.json

```
{
  "competition-data-dir": "seizure-data",
  "data-cache-dir": "data-cache",
  "submission-dir": "./submissions"
}
```

* `competition-data-dir`: directory containing the downloaded competition data
* `data-cache-dir`: directory the task framework will store cached data
* `submission-dir`: directory submissions are written to
