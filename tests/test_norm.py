import numpy as np

from ml.norm import (column_mean, column_std, feature_normalize,
                     add_column_of_ones_to_matrix)
from .utils import MatrixTestCase


class ColumnMeanTest(MatrixTestCase):
    def test_1x3_matrix(self):
        X = np.matrix('1  2  3')
        mean = column_mean(X)
        expected_mean = np.matrix('1  2  3')
        self.assertMatrixEqual(mean, expected_mean)

    def test_2x3_matrix(self):
        X = np.matrix('''
            1  2  3;
            6  4  2
        ''')
        mean = column_mean(X)
        expected_mean = np.matrix('3.5  3  2.5')
        self.assertMatrixEqual(mean, expected_mean)

    def test_column_vector(self):
        X = np.matrix('''
            1;
            2;
            3
        ''')
        mean = column_mean(X)
        expected_mean = np.matrix('2')
        self.assertMatrixEqual(mean, expected_mean)


class ColumnStdTest(MatrixTestCase):
    def test_2x3_matrix(self):
        X = np.matrix('''
            1  2  3;
            6  4  2
        ''')
        std = column_std(X)
        expected_std = np.matrix('3.5355  1.4142  0.7071')
        self.assertMatrixAlmostEqual(std, expected_std)

    def test_column_vector(self):
        X = np.matrix('''
            1;
            2;
            3
        ''')
        std = column_std(X)
        expected_std = np.matrix('1')
        self.assertMatrixAlmostEqual(std, expected_std)


class FeatureNormalizeTest(MatrixTestCase):
    def test_3x3_matrix(self):
        X = np.matrix('''
            8  1  6;
            3  5  7;
            4  9  2
        ''')
        X_norm = feature_normalize(X, column_mean(X), column_std(X))
        expected_X_norm = np.matrix('''
             1.13389  -1   0.37796;
            -0.75593   0   0.75593;
            -0.37796   1  -1.13389
        ''')
        self.assertMatrixAlmostEqual(X_norm, expected_X_norm)

    def test_4x4_matrix(self):
        X = np.matrix('''
             1  2  3  1;
             6  4  2  0;
            11  3  3  9;
             4  9  8  8
        ''')
        X_norm = feature_normalize(X, column_mean(X), column_std(X))
        expected_X_norm = np.matrix('''
            -1.07062  -0.80408  -0.36927  -0.75192;
             0.11896  -0.16082  -0.73855  -0.96676;
             1.30854  -0.48245  -0.36927   0.96676;
            -0.35687   1.44735   1.47710   0.75192
        ''')
        self.assertMatrixAlmostEqual(X_norm, expected_X_norm)

    def test_column_vector(self):
        X = np.matrix('''
            -15.9368;
            -29.1530;
             36.1895;
             37.4922;
            -48.0588;
             -8.9415;
             15.3078;
            -34.7063;
              1.3892;
            -44.3838;
              7.0135;
             22.7627
        ''')
        X_norm = feature_normalize(X, column_mean(X), column_std(X))
        expected_X_norm = np.matrix('''
            -0.36214;
            -0.80320;
             1.37747;
             1.42094;
            -1.43415;
            -0.12869;
             0.68058;
            -0.98853;
             0.21608;
            -1.31150;
             0.40378;
             0.92938
        ''')
        self.assertMatrixAlmostEqual(X_norm, expected_X_norm)


class AddColumnOfOnesToMatrixTest(MatrixTestCase):
    def test_3x3_matrix(self):
        X = np.matrix('''
            2  3  4;
            2  3  4
        ''')
        X_with_ones = add_column_of_ones_to_matrix(X)
        expected_X_with_ones = np.matrix('''
            1  2  3  4;
            1  2  3  4
        ''')
        self.assertMatrixEqual(X_with_ones, expected_X_with_ones)
