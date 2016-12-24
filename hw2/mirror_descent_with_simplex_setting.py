import math
import numpy as np
from hw2 import abstract_search_method
from hw2 import step_size


class MirrorDescentMethod(abstract_search_method.SearchMethod):
    def __init__(self, search_state, step_size_selector):
        super(MirrorDescentMethod, self).__init__(search_state)
        self._iteration_k = 0
        assert isinstance(step_size_selector, step_size.StepSizeSelector)
        self._step_size_selector = step_size_selector

    def step(self):
        self._iteration_k += 1
        subgradient = self.state().middle_subgradient()
        # L-infinite norm of subgradient
        subgradient_norm = np.max(np.abs(subgradient))
        t_k = self._step_size_selector.step_size(subgradient_norm, self.state().score(),
                                                 self._iteration_k)
        shifted = t_k * subgradient - (1. + np.log(self.state().x()))

        argmin_penalized_approximation = softmax(-shifted)
        self._state = self.state().move_to_x(argmin_penalized_approximation)


def softmax(x):
    """
    Calculates softmax function:
        softmax_i = e ** x_i / SUM_j(e ** x_j)
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


