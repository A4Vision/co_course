from l1_projection import *
import numpy as np
from hw2 import abstract_search_method
from hw2 import sfista_method

class DynamicSFISTAMethod(abstract_search_method.SearchMethod):
    """
    SFISTA implemented to solve the smoothed problem:
    min [(sum (huber(a_i * x - b_i)) + sumplex_indicator(x)]
    """

    def __init__(self, search_state, initial_mu, reduction_interval=20, rate=2, assaf=1.0):
        """
        mu - smoothing parameter
        L - An upper bound on the Lipschitz constant of grad(f).
        """
        super(DynamicSFISTAMethod, self).__init__(search_state)
        self._iteration_k = 0
        self._mu = initial_mu
        self._huber_calc = sfista_method.HuberCalculator(self._mu)
        self._L = sfista_method.calculate_L_f(self._state.A(), self._mu)
        self._y_k = self._state.x()
        self._reduction_interval = reduction_interval
        self._rate = rate
        self._t_k = 1
        self._assaf = assaf

    def step(self):
        self._iteration_k += 1
        if self._iteration_k % self._reduction_interval == 0:
            self._mu /= self._rate
            self._L *= sfista_method.calculate_L_f(self._state.A(), self._mu)
            self._huber_calc = sfista_method.HuberCalculator(self._mu)
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
        res = sum(self._huber_calc.huber_derivative(x, self._state.A()[i,:], self._state.b()[i]) for i in range(0, self._state.A().shape[0]))
        return res

    def get_next_t(self, current_t):
        return (1 + (1 + 4 * (current_t ** 2)) ** 0.5) / 2

    def get_next_y(self, current_x, last_x, new_t, current_t):
        return current_x + self._assaf * (current_t - 1.0) / new_t * (current_x - last_x)