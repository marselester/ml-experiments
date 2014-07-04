import numpy as np

from .split import split_to_sets, split_feature_from_target
from .norm import (column_mean, column_std, feature_normalize,
                   add_column_of_ones_to_matrix)

csvfile_path = 'data/concrete.csv'
dataset = np.genfromtxt(csvfile_path, delimiter=',')

# Split dataset.
train_set, cv_set, test_set = split_to_sets(dataset)
X_train, y_train = split_feature_from_target(train_set)
X_cv, y_cv = split_feature_from_target(cv_set)
X_test, y_test = split_feature_from_target(test_set)

# Normalize features.
mean = column_mean(X_train)
std = column_std(X_train)
X_norm_train = feature_normalize(X_train, mean, std)
X_norm_cv = feature_normalize(X_cv, mean, std)
X_norm_test = feature_normalize(X_test, mean, std)

# Add intercept term to feature matrices.
X_norm_train = add_column_of_ones_to_matrix(X_norm_train)
X_norm_cv = add_column_of_ones_to_matrix(X_norm_cv)
X_norm_test = add_column_of_ones_to_matrix(X_norm_test)
