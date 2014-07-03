from __future__ import division

import numpy as np


def hypothesis(X, theta):
    """Predicts target variable.

    :param X: Row vector which contains features.
    :param theta: Column vector that contains parameters of a model.

    """
    return X * theta


def cost(X, y, theta):
    """Returns cost (how close hypothesis is to target variable)."""
    examples_count, _ = X.shape
    h = hypothesis(X, theta)
    sum_ = np.sum(
        np.power(h - y, 2)
    )
    return sum_ / (2 * examples_count)


def derivative_of_cost(X, y, theta):
    """Returns derivative of cost function."""
    examples_count, _ = X.shape
    h = hypothesis(X, theta)
    return (X.transpose() * (h - y)) / examples_count


def predict(X):
    """We must first normalize x using the mean and standard deviation
    that we had previously computed from the training set.

    """
