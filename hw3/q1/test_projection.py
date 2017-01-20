import unittest

import numpy as np

import projection
from hw3.q1 import q1_search_state


class TestProjection(unittest.TestCase):
    def _l2_square_distance(self, (x0, y0, z0), (x, y)):
        return np.linalg.norm(np.array([x0, y0, z0]) - np.array([x, y, x ** 2 + y ** 2])) ** 2

    def test_parabloid_projection(self):
        for x0, y0, z0 in [(1, 2, 3), (4, 5, 6), (4, 5, 2), (1, 0, 0.5), (0, 1, 0.5)]:
            point0 = (x0, y0, z0)
            print 'point0', point0
            x, y, z = projection.project_to_parabloid_epigraph(x0, y0, z0)
            self.assertAlmostEqual(x ** 2 + y ** 2, z)
            epsilon = 1e-5
            distance = self._l2_square_distance(point0, (x, y))
            A = x ** 2 + y ** 2 - z0
            print 'differential:'
            print 2 * (x - x0) + 2 * A * x * 2
            print 2 * (y - y0) + 2 * A * y * 2
            for other_xy in [(x + epsilon, y), (x - epsilon, y), (x, y - epsilon), (x, y + epsilon)]:
                self.assertLess(distance, self._l2_square_distance(point0, other_xy))

    def test_two_parabloid_projection(self):
        np.random.seed(123)
        for _ in xrange(100):
            rand = np.random.random(size=5)
            state = q1_search_state.Q1State(*rand.tolist())
            next_state = projection.project_to_parabloids_intersection(state)
            x0, y0, z0 = next_state.x1, next_state.x2, next_state.y1
            x, y, z = projection.project_to_parabloid_epigraph(x0, y0, z0)
            self.assertAlmostEqual(x0, x)
            self.assertAlmostEqual(y0, y)
            self.assertAlmostEqual(z0, z)
