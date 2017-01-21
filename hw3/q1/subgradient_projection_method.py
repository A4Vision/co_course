from scipy.spatial import distance

from hw3.q1 import abstract_search_method
from hw3.q1 import q1_search_state
from hw3.q1 import step_size
from hw3.q1 import projection
from hw3.q1 import single_variable_state


class SubgradientProjectionMethod(abstract_search_method.SearchMethod):
    def __init__(self, search_state, step_size_selector, use_naive):
        assert isinstance(search_state, q1_search_state.Q1State) or isinstance(search_state, single_variable_state.SingleVarState)
        assert isinstance(step_size_selector, step_size.StepSizeSelector)
        super(SubgradientProjectionMethod, self).__init__(search_state)
        self._step_size_selector = step_size_selector
        self._iteration_k = 0
        self._use_naive = use_naive

    def step(self, eta):
        self._iteration_k += 1
        if self._use_naive:
            subgradient = self.state().subgradient_naive()
        else:
            subgradient = self.state().subgradient_vec()
        shifted = self._state.as_vec() - eta * subgradient
        if self._use_naive:
            projected = self._state.from_vec(shifted)
        else:
            projected = projection.project_to_parabloids_intersection(q1_search_state.Q1State.from_vec(shifted))
        self._state = projected

