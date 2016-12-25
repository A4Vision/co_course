__author__ = 'Amit Botzer'

from hw2 import dynamic_sfista_method
import unittest
import random_problem
import numpy as np


class TestSFISTAMethod(unittest.TestCase):

    def setUp(self):
        A = np.array([[1, -1.5, 1.5, 2, -0.5], [1, -1.5, 1.5, 2, -0.5], [1, -1.5, 1.5, 2, -0.5],
                     [1, -1.5, 1.5, 2, -0.5], [1, -1.5, 1.5, 2, -0.5]])
        self.x = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        b = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
        problem = random_problem.Problem(A, b)
        search_state = random_problem.SearchState(problem, self.x)
        self.dynamic_sfista_method = dynamic_sfista_method.DynamicSFISTAMethod(search_state, 25, 5000, 2, 10)

    def test_sfista_sanity_check(self):
        xs = []
        for i in range(10):
            self.dynamic_sfista_method.step()
            xs.append(self.dynamic_sfista_method._state.x())
        print xs