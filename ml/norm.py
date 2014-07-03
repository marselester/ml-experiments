import numpy as np


def column_mean(X):
    """Returns the average of the matrix elements along columns."""
    return np.apply_over_axes(np.mean, X, axes=0)


def column_std(X):
    """Returns the standard deviation of the matrix elements along columns."""
    return np.std(X, axis=0, ddof=1)


def feature_normalize(X, mean, std):
    return (X - mean) / std
