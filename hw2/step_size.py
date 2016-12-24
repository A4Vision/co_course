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


class SmallerThanOtherSelector(StepSizeSelector):
    def __init__(self, step_size_selector, factor):
        super(SmallerThanOtherSelector, self).__init__()
        self._underlying_selector = step_size_selector
        self._factor = factor

    def step_size(self, subgradient_norm, current_score, iteration_k):
        return self._factor * self._underlying_selector.step_size(subgradient_norm, current_score, iteration_k)


# Maximal alpha s.t. for all x,y: Bergman(x, y) >= L1-norm(x - y) ** 2
SIMPLEX_L1_ALPHA = 1.


class MirrorDescentSimplexStepSizeSelector(StepSizeSelector):
    def __init__(self, theta):
        super(MirrorDescentSimplexStepSizeSelector, self).__init__()
        assert theta > 0
        self._theta = theta

    def step_size(self, subgradient_norm, current_score, iteration_k):
        return math.sqrt(self._theta * SIMPLEX_L1_ALPHA / iteration_k) / subgradient_norm


def strict_theta(n):
    """
    Struct upper bound for Bergman(x_true, x1), given that:
        x1 = (1 / n, ... 1/n)
        x_true is randomized uniformly.
    :param n:
    :return:
    """
    return math.log(n)


def empirical_theta(n):
    """
    Empirical upper bound for Bergman(x_true, x1), given that:
        x1 = (1 / n, ... 1/n)
        x_true is randomized uniformly.
    :param n:
    :return:
    """
    x = np.random.uniform(0, 1, size=(n, 1000))
    x /= np.sum(x, axis=0)
    # EMPIRICAL_MAX(SUM_i(x_i * log(x_i))
    weight = np.sum(x * np.log(x), axis=0)
    return np.max(weight) + math.log(n)
