from l1_projection import *
import numpy as np
from hw2 import abstract_search_method
import math


def calculate_mu(epsilon, m):
    return epsilon / m


def calculate_L_f(A, mu):
    return np.linalg.norm(A, 2) / mu


class HuberCalculator(object):

    def __init__(self, mu):
        self._mu = mu

    def huber_derivative(self, x, a, b):
        """
        Computes the derivative of f(z)=huber(a*x-b) where:
        a - R^n row vector
        x - R^n column vector
        b - real number
        """
        reshaped_x = x.reshape(a.shape[0])
        z = np.dot(a, reshaped_x) - b
        if abs(z) <= self._mu:
            return z / self._mu * a
        elif abs(z) > self._mu:
            return np.sign(z) * a


class SFISTAMethod(abstract_search_method.SearchMethod):
    """
    SFISTA implemented to solve the smoothed problem:
    min [(sum (huber(a_i * x - b_i)) + sumplex_indicator(x)]
    """

    def __init__(self, search_state, mu, L):
        """
        mu - smoothing parameter
        L - An upper bound on the Lipschitz constant of grad(f).
        """
        super(SFISTAMethod, self).__init__(search_state)
        self._iteration_k = 0
        self._huber_calc = HuberCalculator(mu)
        self._L = L
        self._y_k = self._state.x()
        self._t_k = 1

    def step(self):
        self._iteration_k += 1
        last_x_k = self._state.x()
        last_t_k = self._t_k
        self._state = self._state.move_to_x(self.get_next_x(self._y_k))
        self._t_k = self.get_next_t(self._t_k)
        self._y_k = self.get_next_y(self._state.x(), last_x_k, self._t_k, last_t_k)

    def get_next_x(self, y):
        to_project = y - 1.0/self._L * self.grad_f(y)
        reshaped = to_project.reshape((y.shape[0]))
        return project_into_simplex(reshaped)

    def grad_f(self, x):
        return sum(self._huber_calc.huber_derivative(x, self._state.A()[i,:], self._state.b()[i]) for i in range(0, self._state.A().shape[0]))

    def get_next_t(self, current_t):
        return (1 + math.sqrt(1 + 4 * (current_t ** 2))) / 2

    def get_next_y(self, current_x, last_x, new_t, current_t):
        return current_x + (current_t - 1.0) / new_t * (current_x - last_x)