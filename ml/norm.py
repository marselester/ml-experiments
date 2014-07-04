from __future__ import division

import numpy as np


def column_mean(X):
    """Returns the average of the matrix elements along columns."""
    return np.apply_over_axes(np.mean, X, axes=0)


def column_std(X):
    """Returns the standard deviation of the matrix elements along columns."""
    return np.std(X, axis=0, ddof=1)


def feature_normalize(X, mean, std):
    """Scales features and set them to zero mean."""
    return (X - mean) / std


def add_column_of_ones_to_matrix(X):
    """Adds column vector of ``1`` to matrix for convenience of notation."""
    rows_num, _ = X.shape
    column_of_ones = np.ones((rows_num, 1))
    return np.hstack((column_of_ones, X))
