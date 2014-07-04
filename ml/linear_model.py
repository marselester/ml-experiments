from __future__ import division

import numpy as np


def hypothesis(X, theta):
    """Predicts target variable.

    :param X: Matrix of features.
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


def batch_gradient_descent(X, y, theta, alpha=0.01):
    """Minimizes a cost function using batch gradient descent.

    ``alpha`` is gradient descent step.

    """
    J = []
    for _ in xrange(1500):
        J.append(cost(X, y, theta))
        # Simultaneously update all thetas.
        theta -= alpha * derivative_of_cost(X, y, theta)
    return theta, J
