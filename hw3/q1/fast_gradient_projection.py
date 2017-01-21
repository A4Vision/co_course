import math
from hw3.q1 import abstract_search_method
from hw3.q1 import q1_search_state
from hw3.q1 import projection


class FastGradientProjectionMethod(abstract_search_method.SearchMethod):
    def __init__(self, search_state, L):
        super(FastGradientProjectionMethod, self).__init__(search_state)
        assert isinstance(search_state, q1_search_state.Q1State)
        self._tk = 1.
        self._L = L
        self._yk = search_state
        self._xk = search_state

    def step(self, eta):
        subgradient = self._yk.subgradient_vec() * eta
        shifted = self._yk.as_vec() - 1. / self._L * subgradient
        prev_xk = self._xk
        self._xk = projection.project_to_parabloids_intersection(q1_search_state.Q1State.from_vec(shifted))
        prev_tk = self._tk
        self._tk = (1. + math.sqrt(1 + 4 * self._tk ** 2)) / 2.
        self._yk = q1_search_state.Q1State.from_vec(self._xk.as_vec() +
                                                    (prev_tk - 1) / self._tk * (self._xk.as_vec() - prev_xk.as_vec()))
        self._state = self._xk

