import numpy as np
import collections

Problem = collections.namedtuple("Problem", ("A", "b"))


def randomize_problem(m=80, n=100):
    np.random.seed(318)
    A = np.random.randn(m, n)
    x_true = np.random.random(size=n)
    x_true /= np.sum(x_true)
    b = np.dot(A, x_true)
    return Problem(A, b), x_true


def l1_norm(x):
    return np.sum(np.abs(x))


class SearchState(object):
    def __init__(self, problem, x0):
        self._x = x0
        self._problem = problem

    def x(self):
        return self._x

    def problem(self):
        return self._problem

    def A(self):
        return self._problem.A

    def b(self):
        return self._problem.b

    def score(self):
        return l1_norm(np.dot(self.A(), self.x()) - self.b())

    def set_x(self, x):
        self._x = x
