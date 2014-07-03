import numpy as np

from ml.linear_model import hypothesis, cost, derivative_of_cost
from .utils import MatrixTestCase


class HypothesisTest(MatrixTestCase):
    def test_1x3_matrix(self):
        X = np.matrix('1  2  3')
        theta = np.matrix('''
            2;
            4;
            6
        ''')
        y = hypothesis(X, theta)
        expected_y = np.matrix('28')
        self.assertMatrixEqual(y, expected_y)

    def test_2x3_matrix(self):
        X = np.matrix('''
            1  2  3;
            4  5  6
        ''')
        theta = np.matrix('''
            2;
            4;
            6
        ''')
        y = hypothesis(X, theta)
        expected_y = np.matrix('''
            28;
            64
        ''')
        self.assertMatrixEqual(y, expected_y)


class CostTest(MatrixTestCase):
    def test_4x2_matrix(self):
        X = np.matrix('''
            1  2;
            1  3;
            1  4;
            1  5
        ''')
        y = np.matrix('''
            7;
            6;
            5;
            4
        ''')
        theta = np.matrix('''
            0.1;
            0.2
        ''')
        J = cost(X, y, theta)
        expected_J = np.float64('11.9450')
        self.assertAlmostEqual(J, expected_J)

    def test_4x3_matrix(self):
        X = np.matrix('''
            1  2  3;
            1  3  4;
            1  4  5;
            1  5  6
        ''')
        y = np.matrix('''
            7;
            6;
            5;
            4
        ''')
        theta = np.matrix('''
            0.1;
            0.2;
            0.3
        ''')
        J = cost(X, y, theta)
        expected_J = np.float64('7.0175')
        self.assertAlmostEqual(J, expected_J)


class DerivativeOfCostTest(MatrixTestCase):
    def test_4x2_matrix(self):
        X = np.matrix('''
            1  5;
            1  2;
            1  4;
            1  5
        ''')
        y = np.matrix('''
            1;
            6;
            4;
            2
        ''')
        theta = np.matrix('''
            0;
            0
        ''')
        derivative = derivative_of_cost(X, y, theta)
        expected_derivative = np.matrix('''
            -3.2500;
            -10.7500
        ''')
        self.assertMatrixEqual(derivative, expected_derivative)
