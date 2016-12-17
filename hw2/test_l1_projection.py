import numpy as np
from scipy.spatial import distance
import l1_projection
import unittest
import logging



print np.equal(np.arange(10), np.arange(10))


def e_i(i, n, dtype=np.float32):
    """
    Returns the i unit vector.

    :param i:
    :param n:
    :param dtype:
    :return:
    """

    x = np.zeros(n, dtype=dtype)
    x[i] = 1
    return x


def add_ei(x, i, epsilon):
    """
    x + e_i * epsilon
    :param i:
    :param epsilon:
    :param x:
    :return:
    """
    print x, i, e_i(i, len(x))
    return x + e_i(i, len(x)) * epsilon


def adjacent_point_in_simplex(x):
    """
    Returns a point on the simplex (SUM(x_i) == 1)
    that is "close" to x.
    :param x:
    :return:
    """
    assert np.all(x >= 0)
    return x / np.sum(x)


class TestSimplexProjection(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_projected_point_is_closer_than_adjacent_points(self):
        n = 5
        original = np.abs(np.random.randn(n) / n)
        projection = l1_projection.project_into_simplex(original)

        for i in xrange(n):
            adjacent = add_ei(original, i, n)
            in_l1_ball = adjacent_point_in_simplex(adjacent)
            self.assertLess(distance.euclidean(original, projection),
                            distance.euclidean(original, in_l1_ball))

    def test_projected_point_is_in_simplex(self):
        for n in xrange(1, 20):
            original = np.abs(np.random.randn(n) * 0.02 + 1. / n)
            projection = l1_projection.project_into_simplex(original)
            self.assertTrue(np.all(projection >= 0))
            self.assertAlmostEqual(np.sum(projection), 1, delta=0.000001)

    def test_projection_point_is_closer_than_adjacent_points_when_point_contains_repeating_values(self):
        n = 5
        T = 100
        for _ in xrange(T):
            original = np.random.randn(n) * 0.2 + 2
            print original.shape, original
            original[2] = original[3] = original[4]
            projection = l1_projection.project_into_simplex(original)
            for i in xrange(n):
                adjacent = add_ei(original, i, n)
                in_l1_ball = adjacent_point_in_simplex(adjacent)
                self.assertLess(distance.euclidean(original, projection),
                                distance.euclidean(original, in_l1_ball))

    def test_projection_is_identity_for_points_in_the_simplex(self):
        for n in xrange(1, 20):
            original = np.random.randn(n) * 0.1 + 0.5
            in_simplex = adjacent_point_in_simplex(original)
            projection = l1_projection.project_into_simplex(in_simplex)
            self.assertTrue(np.all(np.isclose(projection, in_simplex)), msg="projection={} in_simplex={}, diff={}".format(projection, in_simplex, projection - in_simplex))



