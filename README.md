# Seizure Detection

This repository contains the submission made by members of the Learning Machines group at the Swedish Institute of
Computer Science for the American Epilepsy Society Seizure Prediction Challenge on Kaggle.

http://www.kaggle.com/c/seizure-prediction

Parts of the code is based on https://github.com/MichaelHills/seizure-detection by Michael Hills for the previous
seizure detection challenge.

This README modelled on https://www.kaggle.com/wiki/ModelSubmissionBestPractices and
https://github.com/MichaelHills/seizure-detection/blob/master/README.md

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

The directory name of the data should match the value in SETTINGS.json under the key `TRAIN_DATA_PATH`.

Then simply run:
```
./train.py
```

This will first extract the features for each subject in the `FEATURE_PATH` directory.

This is a long-running process and can take many hours to finish. Make sure to update the `WORKERS`
parameter in `SETTINGS.json` to increase the number of feature extraction jobs run in parallel.

One classifier is then trained for each subject, and dumped to the `FEATURE_PATH\[Subject]` directory. The classifier filename
will also include the timestamp of when the classifier was created.

pp```
FEATURE_PATH/Dog_1/model_svm_[TIMESTAMP].pickle
FEATURE_PATH/Dog_2/model_svm_[TIMESTAMP].pickle
...
FEATURE_PATH/Patient_2/model_svm_[TIMESTAMP].pickle
```

This is also a long running process that can take more than 2 hours to finish. It also requires a high amount of
memory, 16GB of RAM is recommended, 8GB will not suffice.

A submission file will be created under the directory specified by the `SUBMISSION_PATH` key
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
