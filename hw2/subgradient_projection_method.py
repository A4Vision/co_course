import math
from hw2 import l1_projection
from scipy.spatial import distance
from hw2 import step_size
from hw2 import abstract_search_method


class SubgradientProjectionMethod(abstract_search_method.SearchMethod):
    def __init__(self, search_state, step_size_selector):
        super(SubgradientProjectionMethod, self).__init__(search_state)
        assert isinstance(step_size_selector, step_size.StepSizeSelector)
        self._step_size_selector = step_size_selector
        self._iteration_k = 0

    def step(self):
        self._iteration_k += 1
        subgradient = self.state().middle_subgradient()
        subgradient_norm = distance.norm(subgradient, ord=2)
        t_k = self._step_size_selector.step_size(subgradient_norm,
                                                 self._state.score(), self._iteration_k)
        shifted = self.state().x() - t_k * subgradient
        projected = l1_projection.project_into_simplex(shifted)
        self._state = self.state().move_to_x(projected)

