import math
import numpy as np

# 0.5 * MAX(L2-norm(x - y) ** 2) for  x, y in simplex.
L1_SIMPLEX_SQUARE_DIAMETER = 2
SGP_THETA = 0.5 * L1_SIMPLEX_SQUARE_DIAMETER


class StepSizeSelector(object):
    def step_size(self, subgradient_norm, current_score, iteration_k):
        raise NotImplementedError


class ConstantStepSize(StepSizeSelector):
    def __init__(self, total_iteration_N):
        super(ConstantStepSize, self).__init__()
        self._N = total_iteration_N

    def step_size(self, subgradient_norm, current_score, iteration_k):
        s_k = math.sqrt(2 * SGP_THETA / self._N)
        return s_k / subgradient_norm ** 2


class DynamicStepSize(StepSizeSelector):
    def step_size(self, subgradient_norm, current_score, iteration_k):
        s_k = math.sqrt(2 * SGP_THETA / iteration_k)
        return s_k / subgradient_norm ** 2


class OptimalStepKnownTargetValue(StepSizeSelector):
    def __init__(self, optimal_value_function):
        super(OptimalStepKnownTargetValue, self).__init__()
        self._optimal_value = optimal_value_function

    def step_size(self, subgradient_norm, current_score, iteration_k):
        diff = current_score - self._optimal_value
        assert diff >= 0
        return diff / subgradient_norm ** 2

