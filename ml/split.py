from __future__ import division


def split_feature_from_target(dataset):
    """Splits dataset to feature and target matrices.

    It expects target to be in the latest column.

    """
    target_col_index = -1
    feature_matrix = dataset[:, :target_col_index]
    target_column_vector = dataset[:, target_col_index]
    return feature_matrix, target_column_vector


def split_to_sets(dataset, cv_rate=20, test_rate=20):
    """Splits dataset to training (60%), cross validation (20%) and
    test sets (20%).
    """
    cols_num, _ = dataset.shape
    num_in_percent = cols_num / 100
    test_num = test_rate * num_in_percent
    cv_num = cv_rate * num_in_percent

    test_set = dataset[:test_num]
    cv_set = dataset[test_num: test_num + cv_num]
    train_set = dataset[test_num + cv_num:]
    return train_set, cv_set, test_set
