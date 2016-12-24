import unittest
from hw2 import search_method_evaluator


class TestMetricsAverage(unittest.TestCase):
    def test_average(self):
        m1 = search_method_evaluator.SearchMetrics([1, 1], [2, 3], [3, 4], [4, 5])
        m2 = search_method_evaluator.SearchMetrics([2, 3], [3, 5], [4, 6], [5, 7])
        expected = search_method_evaluator.SearchMetrics([1.5, 2], [2.5, 4], [3.5, 5], [4.5, 6])
        actual = search_method_evaluator.average_metrics([m1, m2])
        self.assertEqual(expected, actual)
