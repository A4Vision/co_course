__author__ = 'Amit Botzer'

from l1_projection import *
import numpy as np


"""

SFISTA implemented to solve the smoothed problem:

min [(sum (huber(a_i * x - b_i)) + sumplex_indicator(x)]

"""
class SFISTA(object):

    def __init__(self, A, b, mu):
        self._A = A
        self._b = b
        self._mu = mu

    """
    x0 - initial solution.
    L - An upper bound on the Lipschitz constant of grad(f).
    """
    def run(self, x0, L, num_of_iterations=100):

        # initialization:
        y = [0] * (num_of_iterations + 1)
        x = [0] * (num_of_iterations + 1)
        t = [0] * (num_of_iterations + 1)
        y[1] = x0
        t[1] = 1

        for k in range(1, num_of_iterations + 1, 1):
            x[k] = self.get_next_x(y[k], L)
            t[k+1] = self.get_next_t(t[k])
            y[k+1] = self.get_next_y(x[k], x[k-1], t[k], t[k+1])

    def get_next_x(self, y, L):
        return project_into_simplex(y - 1/L * self.grad_f(y))

    def grad_f(self, x):
        return sum(self.huber_derivative(x, self._A[i,:], self._b[i]) for i in range(1, self._A.shape[0]))

    def get_next_t(self, current_t):
        return (1 + (1 + 4 * (current_t ** 2)) ** 0.5) / 2

    def get_next_y(self, current_x, last_x, current_t, new_t):
        return current_x + (current_t - 1) / new_t * (current_x - last_x)

    def smoothed_f(self, x):
        return sum(self.huber(np.dot(self._A[i,:], x) - self._b[i]) for i in range(1, self._A.shape[0]))

    def huber(self, z):
        if abs(z) <= self._mu:
            return z ** 2 / (2 * self._mu)
        else:
            return abs(z) - self._mu / 2

    """
    Computes the derivative of f(z)=huber(a*x-b) where:
    a - R^n row vector
    x - R^n column vector
    b - real number
    """
    def huber_derivative(self, x, a, b):
        if abs(np.dot(a, x) - b) < self._mu:
            return (np.dot(a, x) - b) / self._mu * a
        else:
            return a
