from __future__ import print_function

import sys
import unittest

import numpy

import blur


class BlurTest(unittest.TestCase):
    def testBlurDistanceIsSmall(self):
        n = 40
        A, b, x = blur.blur(n, 3, 0.8)
        diff = A * x - b
        sum_squares_diff = numpy.sum(diff ** 2) / (n ** 2)
        sum_squares_b = numpy.sum(b ** 2) / (n ** 2)
        epsilon = 0.01
        print("square error", sum_squares_b, sum_squares_diff, file=sys.stderr)

        self.assertGreater(epsilon * sum_squares_b, sum_squares_diff)

