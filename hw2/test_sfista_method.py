__author__ = 'Amit Botzer'

import math
from sfista_method import *
import unittest
import random_problem
import numpy as np


class TestHuberCalculator(unittest.TestCase):

    def setUp(self):
        self.huber_calc = HuberCalculator(0.5)

    def test_huber_with_bigger_value_than_mu(self):
        self.assertAlmostEqual(0.01, self.huber_calc.huber(0.1), 7)

    def test_huber_with_smaller_value_than_mu(self):
        self.assertAlmostEqual(0.75, self.huber_calc.huber(1), 7)

    def test_huber_with_mu(self):
        self.assertAlmostEqual(0.25, self.huber_calc.huber(0.5), 7)

    def test_huber_derivative_where_obj_smaller_than_mu(self):
        a = np.array([1, -1.5, 1.5, 2, -0.5])
        x = np.array([[0.2], [0.2], [0.2], [0.2], [0.2]])
        b = 0.25
        res = self.huber_calc.huber_derivative(x, a, b)
        self.assertItemsEqual(np.array([0.5, -0.75, 0.75, 1, -0.25]), res)

    def test_huber_derivative_where_obj_grater_than_mu(self):
        a = np.array([2, -3, 3, 4, -1])
        x = np.array([[0.2], [0.2], [0.2], [0.2], [0.2]])
        b = 0.25
        res = self.huber_calc.huber_derivative(x, a, b)
        self.assertItemsEqual(a, res)

    def test_huber_derivative_where_obj_equal_mu(self):
        a = np.array([1, -1.5, 1.5, 2, -0.5])
        x = np.array([[0.2], [0.2], [0.2], [0.2], [0.2]])
        b = 0
        with self.assertRaises(Exception):
            self.huber_calc.huber_derivative(x, a, b)

class TestSFISTAMethod(unittest.TestCase):

    def setUp(self):
        problem, x0 = random_problem.randomize_problem()
        search_state = random_problem.SearchState(problem, x0)
        self.sfista_method = SFISTAMethod(search_state, 0.5, 1)

    def test_get_next_t_where_t_is_one(self):
        self.assertAlmostEqual(1.61803, self.sfista_method.get_next_t(1), 5)

    def test_get_next_t_where_t_is_sqrt_2(self):
        self.assertAlmostEqual(2, self.sfista_method.get_next_t(math.sqrt(2)), 7)