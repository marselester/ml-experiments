from unittest import TestCase


class MatrixTestCase(TestCase):
    def assertMatrixEqual(self, matrix1, matrix2):
        self.assertEqual(
            matrix1.tolist(),
            matrix2.tolist()
        )

    def assertMatrixAlmostEqual(self, matrix1, matrix2, places=4):
        cols_num, rows_num = matrix1.shape
        for col_index in xrange(cols_num):
            for row_index in xrange(rows_num):
                self.assertAlmostEqual(
                    matrix1[col_index, row_index],
                    matrix2[col_index, row_index],
                    places=places
                )
