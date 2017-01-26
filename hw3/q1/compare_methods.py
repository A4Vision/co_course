import collections
import math

import numpy as np
from matplotlib import pyplot as plt

from hw3.q1.methods import fast_gradient_projection, gradient_projection, single_variable_search
from hw3.q1.utils import q1_simple_search_state


def best_value():
    """
    Returns the best possible objective function.
    :return:
    """
    method = single_variable_search.SingleVarSearch(1.5)
    for i in xrange(3000):
        method.step(None)
    return method.full_solution().score()


def Q1_comparison():
    np.random.seed(1234)
    state0_primal = q1_simple_search_state.PrimalState.random_state().projection()
    state0_dual = q1_simple_search_state.DualState.random_state().projection()

    gradient_projection_primal = gradient_projection.GradientProjectionMethod(state0_primal)
    fgp_dual = fast_gradient_projection.FastGradientProjectionMethod(state0_dual)

    scores_primal = collections.defaultdict(list)
    scores = collections.defaultdict(list)
    solutions100 = {}

    for i in xrange(1000):
        for (method, eta) in [
            (gradient_projection_primal, 0.5 / math.sqrt(i + 2)),
            (fgp_dual, None)
        ]:
            scores_primal[method].append(method.full_solution().score_primal())
            scores[method].append(method.full_solution().score())
            method.step(eta)

            if i == 100:
                solutions100[method] = method.full_solution()

    print "Final Solutions:\n=============="
    print "Dual fast gradient projection solution:"
    print solutions100[fgp_dual]
    print "Primal gradient projection solution:"
    print solutions100[gradient_projection_primal]
    print 'FastGradientProjection on dual last score', fgp_dual.full_solution().score_primal()
    print 'GradientProjection last score', gradient_projection_primal.score()
    # Plot required by exercise
    for N in (7, 100):
        plt.figure(figsize=(12, 10))
        plt.title("objective function as function of iteration")
        plt.xlabel("iteration")
        plt.ylabel("h(x_k)")
        plt.plot(scores_primal[gradient_projection_primal][:N], 'r->', label='primal objective function - gradient projection')
        plt.plot(scores_primal[fgp_dual][:N], 'm-*', label='primal objective function - dual fast gradient projection')
        plt.plot(-np.array(scores[fgp_dual][:N]), 'b-+', label='dual objective function - dual fast gradient projection')
        plt.legend()
        plt.savefig("q1c_first_{}_iterations.png".format(N))


    h_best = best_value()
    print 'best solution - found with single variable optimization', h_best
    # Plot required by exercise.

    for N in (100, 1000):
        plt.figure(figsize=(12, 10))
        plt.title("Score minus best_score first {} iterations".format(N))
        plt.xlabel("iteration")
        plt.ylabel("h(x_k) - h*")
        plt.semilogy(np.array(scores_primal[fgp_dual][:N]) - h_best, 'm--', label='dual fast gradient projection')
        plt.semilogy(np.array(scores_primal[gradient_projection_primal][:N]) - h_best, 'r->', label='gradient projection')
        plt.legend()
        plt.savefig("q1d_first_{}_iterations_diff_from_best.png".format(N))


def main():
    Q1_comparison()


if __name__ == '__main__':
    main()
