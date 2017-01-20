from scipy.spatial import distance

from hw3.q1 import abstract_search_method
from hw3.q1 import q1_search_state
from hw3.q1 import step_size
from hw3.q1 import projection


class SubgradientProjectionMethod(abstract_search_method.SearchMethod):
    def __init__(self, search_state, step_size_selector, use_naive):
        assert isinstance(search_state, q1_search_state.Q1State)
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
        subgradient_norm = distance.norm(subgradient, ord=2)
        # t_k = self._step_size_selector.step_size(subgradient_norm,
        #                                          self._state.score(), self._iteration_k)
        t_k = eta
        shifted = self._state.as_vec() - t_k * subgradient
        if self._use_naive:
            projected = q1_search_state.Q1State.from_vec(shifted)
        else:
            projected = projection.project_to_parabloids_intersection(q1_search_state.Q1State.from_vec(shifted))
        self._state = projected

