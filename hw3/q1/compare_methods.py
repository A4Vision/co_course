import collections
import math

import numpy as np
from matplotlib import pyplot as plt

from hw3.q1.methods import fast_gradient_projection, gradient_projection, single_variable_search, naive_gradient_descent
from hw3.q1.utils import q1_search_state


def best_value():
    """
    Returns the best possible score.
    :return:
    """
    method = single_variable_search.SingleVarSearch(1.5)
    for i in xrange(3000):
        method.step(None)
    return method.full_solution().score()


def Q1_comparison():
    np.random.seed(1234)
    #
    state0 = q1_search_state.Q1State.random_state()

    gradient_projection_5_vars = gradient_projection.GradientProjectionMethod(state0)
    naive_3_vars = gradient_projection.GradientProjectionMethod(state0)
    single_var = single_variable_search.SingleVarSearch(state0.x2)
    fgp = fast_gradient_projection.FastGradientProjectionMethod(state0)

    scores = collections.defaultdict(list)

    for i in xrange(1000):
        for (method, eta) in [(gradient_projection_5_vars, 1. / math.sqrt(i + 2)),
                              (naive_3_vars, 1.5 / math.sqrt(i + 2)),
                              (single_var, None),
                              (fgp,  None)]:
            scores[method].append(method.score())
            method.step(eta)
    plt.figure(figsize=(20, 10))

    plt.plot(scores[gradient_projection_5_vars][:100], 'r->', label='gradient projection')
    plt.plot(scores[naive_3_vars][:100], 'b-<', label='naive gradient descent')
    plt.plot(scores[single_var][:100], 'y-+', label='single variable')
    plt.plot(scores[fgp][:100], 'g-*', label='fast gradient projection')

    plt.legend()
    plt.savefig("first_100_iterations.png")

    h_best = best_value()

    plt.figure(figsize=(20, 10))

    plt.semilogy(np.array(scores[gradient_projection_5_vars]) - h_best, 'r->', label='gradient projection')
    plt.semilogy(np.array(scores[naive_3_vars]) - h_best, 'b-<', label='naive gradient descent')
    # plt.semilogy(np.array(scores[single_var]) - h_best, 'y-+', label='single variable')
    plt.semilogy(np.array(scores[fgp]) - h_best, 'g-*', label='fast gradient projection')

    plt.legend()

    plt.savefig("first_1000_iterations_diff_from_best.png")

    print gradient_projection_5_vars.full_solution(), gradient_projection_5_vars.score()
    print naive_3_vars.full_solution(), naive_3_vars.score()
    print single_var.full_solution(), single_var.score()
    print fgp.full_solution(), fgp.score()


def main():
    Q1_comparison()


if __name__ == '__main__':
    main()
