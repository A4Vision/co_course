import math
import numpy as np
from hw3.q1 import equation_solver
from hw3.q1.q1_search_state import D1, D3
from hw3.q1 import q1_search_state


def argmin(values, func):
    return values[np.argmin(map(func, values))]


class SingleVarSearch(object):
    def __init__(self, state):
        self._state = state
        self._eta = 0.4
        self._beta = 0.9
        self._k = 0

    def step(self, eta):
        x = self._state.x2
        other_x = (x - self._eta * self._beta, x + self._eta)
        next_x = argmin(other_x, lambda y: SingleVarState(y).score())
        self._eta = next_x - self._state.x2
        # print self._eta
        self._state = SingleVarState(next_x)
        self._k += 1

    def state(self):
        return self._state


class SingleVarState(object):
    def __init__(self, x2):
        self.x2 = x2
        self._score = self._calc_score()

    def state(self):
        x1 = self._x1()
        x3 = self._x3()
        return q1_search_state.Q1State(x1, self.x2, x3, 0, 0)

    def _calc_score(self):
        return self.state().score()

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
        solutions = equation_solver.get_real_root_with_sign(coefs, 1, 2)
        x = argmin(solutions, gA)
        assert gA(x + 0.0001) >= gA(x)
        assert gA(x - 0.0001) >= gA(x)
        return x

    def _x1(self):
        return self._minimize_gA(D1, 2)

    def _x3(self):
        return self._minimize_gA(D3, 1)

    def score(self):
        return self._score
