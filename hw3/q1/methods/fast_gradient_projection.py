import math
from hw3.q1.methods import abstract_search_method
from hw3.q1.utils import projection, q1_search_state


class FastGradientProjectionMethod(abstract_search_method.SearchMethod):
    def __init__(self, search_state):
        assert isinstance(search_state, q1_search_state.Q1State)
        self._tk = 1.
        self._L = q1_search_state.Q1State.LIPSCHITZ_L
        self._yk = search_state.as_vec()
        self._xk = search_state.as_vec()

    def step(self, _):
        subgradient = q1_search_state.Q1State.from_vec(self._yk).subgradient_5_variables()
        shifted = self._yk - 1. / self._L * subgradient
        shifted_state = q1_search_state.Q1State.from_vec(shifted)
        prev_xk = self._xk
        self._xk = projection.project_to_parabloids_intersection(shifted_state).as_vec()
        prev_tk = self._tk
        self._tk = (1. + math.sqrt(1 + 4 * self._tk ** 2)) / 2.
        self._yk = self._xk + (prev_tk - 1) / self._tk * (self._xk - prev_xk)

    def full_solution(self):
        return q1_search_state.Q1State.from_vec(self._xk)
