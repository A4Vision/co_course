from scipy.spatial import distance

from hw3.q1 import abstract_search_method
from hw3.q1 import q1_search_state
from hw3.q1 import step_size
from hw3.q1 import projection


class SubgradientProjectionMethod(abstract_search_method.SearchMethod):
    def __init__(self, search_state, step_size_selector):
        assert isinstance(search_state, q1_search_state.Q1State)
        super(SubgradientProjectionMethod, self).__init__(search_state)
        assert isinstance(step_size_selector, step_size.StepSizeSelector)
        self._step_size_selector = step_size_selector
        self._iteration_k = 0

    def step(self):
        self._iteration_k += 1
        subgradient = self.state().subgradient()
        subgradient_norm = distance.norm(subgradient, ord=2)
        t_k = self._step_size_selector.step_size(subgradient_norm,
                                                 self._state.score(), self._iteration_k)
        shifted = self.state().x() - t_k * subgradient
        projected = projection.project_to_parabloids_intersection(shifted)
        self._state = self.state().move_to_x(projected)

