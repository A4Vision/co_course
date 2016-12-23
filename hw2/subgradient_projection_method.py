import math
from hw2 import l1_projection
from scipy.spatial import distance
from hw2 import random_problem
from hw2 import abstract_search_method


L1_SIMPLEX_SQUARE_DIAMETER = 2
THETA = 0.5 * L1_SIMPLEX_SQUARE_DIAMETER


class SubgradientProjectionMethod(abstract_search_method.SearchMethod):
    def __init__(self, search_state, step_size_selector):
        super(SubgradientProjectionMethod, self).__init__(search_state)
        assert isinstance(step_size_selector, StepSizeSelector)
        assert isinstance(search_state, random_problem.SearchState)
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


class StepSizeSelector(object):
    def step_size(self, subgradient_norm, current_score, iteration_k):
        raise NotImplementedError


class ConstantStepSize(StepSizeSelector):
    def __init__(self, total_iteration_N):
        super(ConstantStepSize, self).__init__()
        self._N = total_iteration_N

    def step_size(self, subgradient_norm, current_score, iteration_k):
        s_k = math.sqrt(2 * THETA / self._N)
        return s_k / subgradient_norm ** 2


class DynamicStepSize(StepSizeSelector):
    def step_size(self, subgradient_norm, current_score, iteration_k):
        s_k = math.sqrt(2 * THETA / iteration_k)
        return s_k / subgradient_norm ** 2


class OptimalStepKnownTargetValue(StepSizeSelector):
    def __init__(self, optimal_value_function):
        super(OptimalStepKnownTargetValue, self).__init__()
        self._optimal_value = optimal_value_function

    def step_size(self, subgradient_norm, current_score, iteration_k):
        diff = current_score - self._optimal_value
        assert diff >= 0
        return diff / subgradient_norm ** 2


class SmallerThanOtherSelector(StepSizeSelector):
    def __init__(self, step_size_selector, factor):
        self._underlying_selector = step_size_selector
        self._factor = factor

    def step_size(self, subgradient_norm, current_score, iteration_k):
        return self._factor * self._underlying_selector.step_size(subgradient_norm, current_score, iteration_k)