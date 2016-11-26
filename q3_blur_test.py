from __future__ import print_function
import numpy
import blur
import unittest
import sys


class BlurTest(unittest.TestCase):
    def testBlurImageIsSmooth(self):
        n = 50
        _, x, b = blur.blur(50, 3, 0.7)
        diff = numpy.diff(x)
        sum_squares_diff = numpy.sum(diff ** 2) / (n ** 2)
        sum_squares_b = numpy.sum(b ** 2) / (n ** 2)
        epsilon = 0.1
        print("square smooth", sum_squares_b, sum_squares_diff, file=sys.stderr)
        self.assertGreater(epsilon * sum_squares_b, sum_squares_diff)

    def testBlurDistanceIsSmall(self):
        n = 40
        A, x, b = blur.blur(n, 3, 0.7)
        diff = numpy.dot(A, x) - b
        sum_squares_diff = numpy.sum(diff ** 2) / (n ** 2)
        sum_squares_b = numpy.sum(b ** 2) / (n ** 2)
        epsilon = 0.01
        self.assertGreater(2, 1)
        print("square error", sum_squares_b, sum_squares_diff, file=sys.stderr)
        self.assertGreater(epsilon * sum_squares_b, sum_squares_diff)
