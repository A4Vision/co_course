import math
from hw3.q1.methods import abstract_search_method


class FastGradientProjectionMethod(abstract_search_method.SearchMethod):
    def __init__(self, search_state):
        self._tk = 1.
        self._state_cls = search_state.__class__
        self._L = self._state_cls.LIPSCHITZ_L
        self._yk = search_state.as_vec()
        self._xk = search_state.as_vec()

    def step(self, _):
        gradient = self._state_cls.from_vec(self._yk).gradient()
        shifted = self._yk - 1. / self._L * gradient
        shifted_state = self._state_cls.from_vec(shifted)
        prev_xk = self._xk
        self._xk = shifted_state.projection().as_vec()
        prev_tk = self._tk
        self._tk = (1. + math.sqrt(1 + 4 * self._tk ** 2)) / 2.
        self._yk = self._xk + (prev_tk - 1) / self._tk * (self._xk - prev_xk)

    def full_solution(self):
        return self._state_cls.from_vec(self._xk)
