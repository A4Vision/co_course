import collections
import functools
from matplotlib import pyplot as plt
import numpy as np
from hw2 import random_problem
from hw2 import abstract_search_method
from hw2 import subgradient_projection_method
from hw2 import mirror_descent_with_simplex_setting
from hw2 import step_size
from scipy.spatial import distance

COLORS = ["red", "green", "blue", "black", "yellow"]
# Amount of iterations in every search.
N_steps = 100
# How many problems to randomize, in order to calculate average metrics.
N_runs = 20
# How many steps to ignore when plotting the metrics.
# It is reasonable to ignore the first steps in order to
# observe better the behavior during convergence, because
# normally the first iterations have very large metrics.
FIRST_STEPS_TO_IGNORE_IN_PLOT = 1

# TODO(assaf): Consider refactoring to using a generic metric.
# Amount of metrics - scores, gradients_sizes, etc.
METRICS_AMOUNT = 4
# Each metric is just a list of numbers that are measured after each step.
SearchMetrics = collections.namedtuple("SearchMetrics", ("scores", "gradients_sizes", "x_delta_sizes",
                                                         "distances_to_x_true"))


def average_lists(lists):
    """
    Average each coordinate of the given lists.
    :param lists:
    :return:

    >>> lists = [[0, 1, 2], [1, 2, 3]]
    >>> average_lists(lists)
    [0.5, 1.5, 2.5]
    """
    return list(np.average(np.array(lists, dtype=np.float64), axis=0))


def average_metrics(metrics_list):
    """
    Average metrics of a given list of metrics.
    :param metrics_list:
    :return:
    """
    averages = [average_lists([metrics[i] for metrics in metrics_list]) for i in xrange(METRICS_AMOUNT)]
    return SearchMetrics(*averages)


def plot_metrics(method_name2metrics):
    """
    Plots the metrics of the given methods on a single graph per metric .
    :param method_name2metrics: mapping method-name --> search-metrics
        dict[str, SearchMetrics]
    :return:
    """
    f, (scores_ax, gradients_ax, distances_ax, deltas_ax) = plt.subplots(4, sharex=True)
    f.set_size_inches(15, 10)
    scores_ax.set_title("Score - ||Ax - b||")
    gradients_ax.set_title("Gradient norm - ||gradient(x)||")
    distances_ax.set_title("Distance to real - ||x - x_true||")
    deltas_ax.set_title("Step delta size - ||x - x_previous||")
    assert len(method_name2metrics) <= len(COLORS), "Not enough colors"
    for (method_name, metrics), color in zip(method_name2metrics.iteritems(), COLORS):
        assert isinstance(metrics, SearchMetrics)
        label = method_name
        scores_ax.plot(metrics.scores[FIRST_STEPS_TO_IGNORE_IN_PLOT:], label=label, color=color)
        gradients_ax.plot(metrics.gradients_sizes[FIRST_STEPS_TO_IGNORE_IN_PLOT:], label=label, color=color)
        distances_ax.plot(metrics.distances_to_x_true[FIRST_STEPS_TO_IGNORE_IN_PLOT:], label=label, color=color)
        deltas_ax.plot(metrics.x_delta_sizes[FIRST_STEPS_TO_IGNORE_IN_PLOT:], label=label, color=color)
    scores_ax.legend()
    gradients_ax.legend()
    distances_ax.legend()
    deltas_ax.legend()
    return f


def calculate_metrics(search_method, x_true):
    """
    Runs the given searcher for N_steps, and calculates the metrics for this run.
    :param search_method:
    :param x_true:
    :return:
    """
    assert isinstance(search_method, abstract_search_method.SearchMethod)
    metrics = SearchMetrics([], [], [], [])
    prev_x = search_method.state().x()
    for i in xrange(N_steps):
        search_method.step()
        state = search_method.state()
        metrics.scores.append(state.score())
        metrics.gradients_sizes.append(state.deterministic_gradient_size())
        metrics.distances_to_x_true.append(state.euclidean_distance_to_target(x_true))
        metrics.x_delta_sizes.append(distance.euclidean(state.x(), prev_x))
        prev_x = state.x()

    return metrics


def measure_metrics_for_various_methods(method_name2method_factory, n_runs):
    """
    :param n_runs: Amount of random problems to measure.
    :param method_name2method_factory:
    dict[str, callable]
        f = method_name2method_factory[method_name]
        f(problem, x0) --> SearchMethod instance.
    :return: dict[str, SearchMetrics]
        Average metrics for each search method.
        Averaged over n_runs.
    """
    randomized = [random_problem.randomize_problem(seed=i) for i in xrange(n_runs)]
    method_name2metrics = {}
    for method_name, method_factory in method_name2method_factory.iteritems():
        all_method_metrics = []
        for problem, x_true in randomized:
            x0 = np.ones_like(x_true) / len(x_true)
            search_state = random_problem.SearchState(problem, x0)
            search_method = method_factory(search_state)
            current_metrics = calculate_metrics(search_method, x_true)
            all_method_metrics.append(current_metrics)
        metrics = average_metrics(all_method_metrics)
        method_name2metrics[method_name] = metrics
    return method_name2metrics


def compare_sgp_step_selectors():
    """
    Plots a graph that compares the various SGP step size selection schemes.
    :return:
    """
    step_size_selectors = [step_size.DynamicStepSize(),
                           step_size.ConstantStepSize(N_steps),
                           # We know that for x_true, we have exactly Ax = b.
                           step_size.OptimalStepKnownTargetValue(0),]
    method_name2factory = {}
    for step_size_selector in step_size_selectors:
        method_name2factory[type(step_size_selector).__name__] = functools.partial(
            subgradient_projection_method.SubgradientProjectionMethod,
            step_size_selector=step_size_selector)
    method_name2metrics = measure_metrics_for_various_methods(method_name2factory, N_runs)
    f = plot_metrics(method_name2metrics)
    f.suptitle("Subgradient Projection Method")
    plt.savefig("sgp.png")
    plt.show()


def compare_sgp_and_mirror_descent():
    """
    Compare SGP, with two versions of EMD:
        * one with theta = log(n) - strict bound over Bergman(x_true, x)
        * another with theta = empirical bound over Bergman(x_true, x)
    :return:
    """
    sgp = functools.partial(subgradient_projection_method.SubgradientProjectionMethod,
                            step_size_selector=step_size.OptimalStepKnownTargetValue(0))

    def mirror_descent1(search_state):
        n = len(search_state.x())
        theta = step_size.strict_theta(n)
        step_size_selector = step_size.MirrorDescentSimplexStepSizeSelector(theta)
        return mirror_descent_with_simplex_setting.MirrorDescentMethod(search_state, step_size_selector)

    def mirror_descent2(search_state):
        n = len(search_state.x())
        theta = step_size.empirical_theta(n)
        step_size_selector = step_size.MirrorDescentSimplexStepSizeSelector(theta)
        return mirror_descent_with_simplex_setting.MirrorDescentMethod(search_state, step_size_selector)

    method_name2factory = {"SGP": sgp, "EMD+theta=log(n)": mirror_descent1,
                           "EMD+theta=log(n)+MAX(SUM(x_i*log(x_i)))": mirror_descent2}
    method_name2metrics = measure_metrics_for_various_methods(method_name2factory, N_runs)
    f = plot_metrics(method_name2metrics)
    f.suptitle("SGV vs. Entropic Mirror Descent")
    plt.savefig("emd_and_sgp.png")
    plt.show()


def main():
    compare_sgp_step_selectors()
    compare_sgp_and_mirror_descent()

if __name__ == '__main__':
    main()
