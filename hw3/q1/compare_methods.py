import collections

import math
import numpy as np
from matplotlib import pyplot as plt
from hw3.q1 import q1_search_state
from hw3.q1 import step_size
from hw3.q1 import subgradient_projection_method
from hw3.q1 import single_variable_state
from hw3.q1 import fast_gradient_projection


def best_value(state0):
    method = single_variable_state.SingleVarSearch(single_variable_state.SingleVarState(state0.x2))
    for i in xrange(1000):
        method.step(None)
    return method.state().state().score()


def run_sgp():
    step_size_selector = step_size.DynamicStepSize()
    np.random.seed(1243)
    state0 = q1_search_state.Q1State.random_state()
    gradient_projection_5_vars = subgradient_projection_method.SubgradientProjectionMethod(state0, step_size_selector,
                                                                                           False)
    method_naive_3_vars = subgradient_projection_method.SubgradientProjectionMethod(state0, step_size_selector,
                                                                                    True)
    method_single_var = single_variable_state.SingleVarSearch(single_variable_state.SingleVarState(state0.x2))
    fgp = fast_gradient_projection.FastGradientProjectionMethod(state0, L=4)

    scores = collections.defaultdict(list)
    for i in xrange(2000):
        for (method, eta) in [(gradient_projection_5_vars, 1. / math.sqrt(i + 2)),
                              (method_naive_3_vars, 0.4 / math.sqrt(i + 3)),
                              (method_single_var, None),
                              (fgp,  4 / math.sqrt(i + 3))]:
            method.step(eta)
            scores[method].append(method.state().score())
    plt.figure(figsize=(200, 400))
    h_best = best_value(state0)
    for (method, description, style) in [
        (gradient_projection_5_vars, 'projection', 'r--'),
        # (method_naive_3_vars, 'naive gradient descent', 'b--'),
        # (method_single_var, 'naive single variable', 'y--'),
        (fgp, 'fast gradient projection', 'g-*')
    ]:
        plt.semilogy(np.array(scores[method] - h_best), style, label=description)

    plt.legend()
    plt.show()
    print gradient_projection_5_vars.state().as_vec(), gradient_projection_5_vars.state().score()
    print method_naive_3_vars.state().as_vec(), method_naive_3_vars.state().score()
    print method_single_var.state().state().as_vec(), method_single_var.state().state().score()
    print fgp.state().as_vec(), fgp.state().score()


def main():
    run_sgp()


if __name__ == '__main__':
    main()
