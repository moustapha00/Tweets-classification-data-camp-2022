import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from rampwf.workflows import FeatureExtractorClassifier
import rampwf as rw

# Define the data and the prediction task
problem_title = 'Tweets Multiclass Classification'
_target_column_name = 'target'

# -----------------------------------------------------------------------------
# Predictions type
# -----------------------------------------------------------------------------

# Predictions should be one of 'negative', 'neutral', or 'positive'
_prediction_label_names = ['negative', 'neutral', 'positive']
Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)


# -----------------------------------------------------------------------------
# Worklow element
# -----------------------------------------------------------------------------

# Define the feature extractor and the classifier
workflow = FeatureExtractorClassifier()


# -----------------------------------------------------------------------------
# Score types
# -----------------------------------------------------------------------------

# Define the evaluation metric
score_types = [
    rw.score_types.ROCAUC(name='auc'),
    rw.score_types.Accuracy(name='acc'),
    rw.score_types.NegativeLogLikelihood(name='nll'),
]

# -----------------------------------------------------------------------------
# Training / testing data reader
# ----------------------------------------------------------------------------- 

# Define the input and output files
_train = 'data/train.csv'
_test = 'data/test.csv'
_submission = 'data/submission.csv'


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)

    # for the "quick-test" mode, use less data
    test = os.getenv("RAMP_TEST_MODE", 0)
    if test:
        N_small = 100
        X_df = X_df[:N_small]
        y_array = y_array[:N_small]

    return X_df, y_array

# Define the training and testing data loaders
def get_train_data(path='.'):
    return _read_data(path, _train)

def get_test_data(path='.'):
    return _read_data(path, _test)


# -----------------------------------------------------------------------------
# Cross-validation scheme
# -----------------------------------------------------------------------------

# Define the cross-validation method
def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    return cv.split(X, y)
