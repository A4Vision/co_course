import math
from scipy.spatial import distance
from hw2 import random_problem
from hw2 import abstract_search_method


L1_SIMPLEX_DIAMETER = 1
THETA = 0.5 * L1_SIMPLEX_DIAMETER


class SubgradientProjectionMethod(abstract_search_method.SearchMethod):
    def __init__(self, search_state, step_size_selector):
        super(SubgradientProjectionMethod, self).__init__(search_state)
        assert isinstance(step_size_selector, StepSizeSelector)
        self._step_size_selector = step_size_selector
        self._iteration_k = 0

    def step(self):
        subgradient = self._state.middle_subgradient()
        t_k = self._step_size_selector.step_size(distance.norm(subgradient, ord=2),
                                                 self._state, self._iteration_k)
        self._state.set_x(self._state.x() - t_k * subgradient)


class StepSizeSelector(object):
    def step_size(self, subgradient_norm, search_state, iteration_k):
        raise NotImplementedError


class ConstantStepSize(StepSizeSelector):
    def __init__(self, total_iteration_N):
        self._N = total_iteration_N

    def step_size(self, subgradient_norm, search_state, iteration_k):
        s_k = math.sqrt(2 * THETA / self._N)
        return s_k / subgradient_norm


class DynamicStepSize(StepSizeSelector):
    def step_size(self, subgradient_norm, search_state, iteration_k):
        s_k = math.sqrt(2 * THETA / iteration_k)
        return s_k / subgradient_norm


class OptimalStepKnownTargetValue(StepSizeSelector):
    def __init__(self, optimal_value_function):
        self._optimal_value = optimal_value_function

    def step_size(self, subgradient_norm, search_state, iteration_k):
        assert isinstance(search_state, random_problem.SearchState)
        diff = search_state.score() - self._optimal_value
        assert diff >= 0
        return diff / subgradient_norm ** 2

