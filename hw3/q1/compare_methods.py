import collections
import math

import numpy as np
from matplotlib import pyplot as plt

from hw3.q1.methods import fast_gradient_projection, gradient_projection, single_variable_search, naive_gradient_descent
from hw3.q1.utils import q1_search_state, q1_dual_search_state


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

    state_dual = q1_dual_search_state.Q1DualState.random_state()

    gradient_projection_5_vars = gradient_projection.GradientProjectionMethod(state0)

    gradient_projection_dual = gradient_projection.GradientProjectionMethod(state_dual)
    dual_fgp = fast_gradient_projection.FastGradientProjectionMethod(state_dual)

    naive_3_vars = naive_gradient_descent.NaiveGradientDescent(state0)
    single_var = single_variable_search.SingleVarSearch(state0.x2)
    fgp = fast_gradient_projection.FastGradientProjectionMethod(state0)

    scores_primal = collections.defaultdict(list)
    scores = collections.defaultdict(list)

    for i in xrange(1000):
        for (method, eta) in [(gradient_projection_5_vars, 1. / math.sqrt(i + 2)),
                              (gradient_projection_dual, 0.2 / math.sqrt(i + 2)),
                              (naive_3_vars, 0.5 / math.sqrt(i + 2)),
                              (single_var, None),
                              (dual_fgp, None),
                              (fgp,  None)]:
            scores_primal[method].append(method.full_solution().score_primal())
            scores[method].append(method.full_solution().score())
            method.step(eta)

    print "Final Solutions:\n=============="
    print "Dual fast gradient projection solution:"
    print dual_fgp.full_solution().as_q1_state()
    print "Primal gradient projection solution:"
    print gradient_projection_5_vars.full_solution()

    # Plot required by exercise
    plt.figure(figsize=(20, 10))
    plt.title("Score as function of iteration")
    plt.xlabel("iteration")
    plt.ylabel("h(x_k)")
    plt.plot(scores_primal[gradient_projection_5_vars][:100], 'r->', label='gradient projection')
    plt.plot(scores_primal[dual_fgp][:100], 'm--', label='dual fast gradient projection')
    plt.legend()
    plt.savefig("q1c_first_100_iterations.png")

    # Plot of all methods.
    plt.figure(figsize=(20, 10))
    plt.plot(scores_primal[gradient_projection_5_vars][:100], 'r->', label='gradient projection')
    plt.plot(scores_primal[gradient_projection_dual][:100], 'c->', label='gradient projection dual')
    plt.plot(scores_primal[naive_3_vars][:100], 'b-<', label='naive gradient descent')
    plt.plot(scores_primal[single_var][:100], 'y-+', label='single variable')
    plt.plot(scores_primal[fgp][:100], 'g-*', label='fast gradient projection')
    plt.plot(scores_primal[dual_fgp][:100], 'm--', label='dual fast gradient projection')
    plt.legend()
    plt.savefig("q1c_first_100_iterations_full_plot.png")

    h_best = best_value()

    # Plot required by exercise.
    plt.figure(figsize=(20, 10))
    plt.title("Score minus best_score")
    plt.xlabel("iteration")
    plt.ylabel("h(x_k) - h*")
    plt.semilogy(np.array(scores_primal[dual_fgp]) - h_best, 'm--', label='dual fast gradient projection')
    plt.semilogy(np.array(scores_primal[gradient_projection_5_vars]) - h_best, 'r->', label='gradient projection')
    plt.legend()
    plt.savefig("q1d_first_1000_iterations_diff_from_best_full_plot.png")
    # Plot of all methods.
    plt.figure(figsize=(20, 10))
    plt.semilogy(np.array(scores_primal[gradient_projection_dual]) - h_best, 'c->', label='dual gradient projection')
    plt.semilogy(np.array(scores_primal[dual_fgp]) - h_best, 'm--', label='dual fast gradient projection')
    plt.semilogy(np.array(scores_primal[gradient_projection_5_vars]) - h_best, 'r->', label='gradient projection')
    plt.semilogy(np.array(scores[naive_3_vars]) - h_best, 'b-<', label='naive gradient descent')
    plt.semilogy(np.array(scores[single_var]) - h_best, 'y-+', label='single variable')
    plt.semilogy(np.array(scores_primal[fgp]) - h_best, 'g-*', label='fast gradient projection')
    plt.legend()
    plt.savefig("q1d_first_1000_iterations_diff_from_best_full_plot.png")


def main():
    Q1_comparison()


if __name__ == '__main__':
    main()
