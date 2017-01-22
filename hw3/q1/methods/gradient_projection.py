from hw3.q1.methods import abstract_search_method
from hw3.q1.utils import projection, q1_search_state


class GradientProjectionMethod(abstract_search_method.SearchMethod):
    def __init__(self, search_state):
        self._state = search_state

    def full_solution(self):
        return self._state

    def step(self, eta):
        gradient = self._state.gradient()
        shifted = self._state.as_vec() - eta * gradient
        state_shifted = self._state.from_vec(shifted)
        self._state = state_shifted.projection()
