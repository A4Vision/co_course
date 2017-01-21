from hw3.q1.methods import abstract_search_method
from hw3.q1.utils import q1_search_state


class GradientProjectionMethod(abstract_search_method.SearchMethod):
    def __init__(self, search_state):
        self._state = search_state

    def full_solution(self):
        return self._state

    def step(self, eta):
        subgradient = self._state.subgradient_3_variables()
        shifted = self._state.as_vec() - eta * subgradient
        self._state = q1_search_state.Q1State.from_vec(shifted)
