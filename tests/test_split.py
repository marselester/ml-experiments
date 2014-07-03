import numpy as np

from ml.split import split_feature_from_target, split_to_sets
from .utils import MatrixTestCase


class SplitFeatureFromTargetTest(MatrixTestCase):
    def test_1x3_matrix(self):
        dataset = np.matrix('1  2  3')
        X, y = split_feature_from_target(dataset)
        expected_X = np.matrix('1  2')
        expected_y = np.matrix('3')
        self.assertMatrixEqual(X, expected_X)
        self.assertMatrixEqual(y, expected_y)

    def test_2x3_matrix(self):
        dataset = np.matrix('''
            1  2  3;
            6  4  2
        ''')
        X, y = split_feature_from_target(dataset)
        expected_X = np.matrix('''
            1  2;
            6  4
        ''')
        expected_y = np.matrix('''
            3;
            2
        ''')
        self.assertMatrixEqual(X, expected_X)
        self.assertMatrixEqual(y, expected_y)


class SplitToSetsTest(MatrixTestCase):
    def test_10x2_matrix(self):
        dataset = np.matrix('''
             1   1;
             2   2;
             3   3;
             4   4;
             5   5;
             6   6;
             7   7;
             8   8;
             9   9;
            10  10
        ''')
        train_set, cv_set, test_set = split_to_sets(dataset)
        expected_train_set = np.matrix('''
             5   5;
             6   6;
             7   7;
             8   8;
             9   9;
            10  10
        ''')
        expected_cv_set = np.matrix('''
            3  3;
            4  4
        ''')
        expected_test_set = np.matrix('''
            1  1;
            2  2
        ''')
        self.assertMatrixEqual(train_set, expected_train_set)
        self.assertMatrixEqual(cv_set, expected_cv_set)
        self.assertMatrixEqual(test_set, expected_test_set)

    def test_6x2_matrix(self):
        dataset = np.matrix('''
            1  1;
            2  2;
            3  3;
            4  4;
            5  5;
            6  6
        ''')
        train_set, cv_set, test_set = split_to_sets(dataset)
        expected_train_set = np.matrix('''
            3  3;
            4  4;
            5  5;
            6  6
        ''')
        expected_cv_set = np.matrix('2  2')
        expected_test_set = np.matrix('1  1')
        self.assertMatrixEqual(train_set, expected_train_set)
        self.assertMatrixEqual(cv_set, expected_cv_set)
        self.assertMatrixEqual(test_set, expected_test_set)
