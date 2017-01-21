import math

import numpy as np

from hw3.q1.methods import abstract_search_method
from hw3.q1.utils import equation_solver, q1_search_state
from hw3.q1.utils.q1_search_state import D1, D3


def argmin(values, func):
    return values[np.argmin(map(func, values))]


class SingleVarSearch(abstract_search_method.SearchMethod):
    def __init__(self, x2):
        self.x2 = x2
        self._eta = 0.4
        self._beta = 0.9

    def step(self, _):
        # Either go back eta * beta, or forward eta.
        other_x = (self.x2 - self._eta * self._beta, self.x2 + self._eta)
        next_x = argmin(other_x, lambda y: SingleVarSearch(y).full_solution().score())
        self._eta = next_x - self.x2
        # print self._eta
        self.x2 = next_x

    def _minimize_gA(self, A, d):
        """
        ArgMin {d*(x-A) ** 2 + 2 * sqrt(x ** 2 + B)}
        with:
            B = x2 ** 2
        :param i:
        :return:
        """
        B = self.x2 ** 2
        coefs = [d ** 2, -2 * d ** 2 * A, d ** 2 * (A ** 2 + B) - 1, -2 * d ** 2 * A * B, d ** 2 * A ** 2 * B]

        def gA(x):
            return d * (x - A) ** 2 + 2 * math.sqrt(x ** 2 + B)
        solutions = equation_solver.real_roots(coefs, 1)
        x = argmin(solutions, gA)
        # Asserting x is a local minimum of gA.
        assert gA(x + 0.0001) >= gA(x)
        assert gA(x - 0.0001) >= gA(x)
        return x

    def _x1(self):
        return self._minimize_gA(D1, 2)

    def _x3(self):
        return self._minimize_gA(D3, 1)

    def full_solution(self):
        x1 = self._x1()
        x3 = self._x3()
        return q1_search_state.Q1State(x1, self.x2, x3, 0, 0)



