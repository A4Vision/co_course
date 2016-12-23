import collections
import functools
from matplotlib import pyplot as plt
import numpy as np
from hw2 import random_problem
from hw2 import abstract_search_method
from hw2 import subgradient_projection_method
from scipy.spatial import distance

COLORS = ["red", "green", "blue", "black", "yellow"]
N_steps = 120
N_runs = 20
FIRST_STEP_TO_IGNORE_IN_PLOT = 20

# TODO(assaf): Consider refactoring to using a generic metric.
# Amount of metrics - scores, gradients_sizes, etc.
METRICS_AMOUNT = 4
SearchMetrics = collections.namedtuple("SearchMetrics", ("scores", "gradients_sizes", "x_delta_sizes",
                                                         "distances_to_x_true"))


def average_lists(lists):
    return np.average(np.array(lists, dtype=np.float64), axis=0)


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
    f.suptitle("Subgradient Projection Method")
    scores_ax.set_title("Score - ||Ax - b||")
    gradients_ax.set_title("Gradient norm - ||gradient(x)||")
    distances_ax.set_title("Distance to real - ||x - x_true||")
    deltas_ax.set_title("Step delta size - ||x - x_previous||")
    assert len(method_name2metrics) <= len(COLORS), "Not enough colors"
    for (method_name, metrics), color in zip(method_name2metrics.iteritems(), COLORS):
        assert isinstance(metrics, SearchMetrics)
        label = method_name
        scores_ax.plot(metrics.scores[FIRST_STEP_TO_IGNORE_IN_PLOT:], label=label, color=color)
        gradients_ax.plot(metrics.gradients_sizes[FIRST_STEP_TO_IGNORE_IN_PLOT:], label=label, color=color)
        distances_ax.plot(metrics.distances_to_x_true[FIRST_STEP_TO_IGNORE_IN_PLOT:], label=label, color=color)
        deltas_ax.plot(metrics.x_delta_sizes[FIRST_STEP_TO_IGNORE_IN_PLOT:], label=label, color=color)
    scores_ax.legend()
    gradients_ax.legend()
    distances_ax.legend()
    deltas_ax.legend()
    plt.draw()


def calculate_metrics(search_method, x_true):
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
    :param methods_factories:
    List[f]
        f(problem, x0) --> SearchMethod
    :return:
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


def evaluate_sgp():
    step_size_selectors = [subgradient_projection_method.DynamicStepSize(),
                           subgradient_projection_method.ConstantStepSize(N_steps),
                           subgradient_projection_method.OptimalStepKnownTargetValue(0),
                           subgradient_projection_method.SmallerThanOtherSelector(
                                   subgradient_projection_method.OptimalStepKnownTargetValue(0), 0.5)]
    method_name2factory = {}
    for step_size_selector in step_size_selectors:
        method_name2factory[type(step_size_selector).__name__] = functools.partial(
            subgradient_projection_method.SubgradientProjectionMethod,
            step_size_selector=step_size_selector)
    method_name2metrics = measure_metrics_for_various_methods(method_name2factory, N_runs)
    plot_metrics(method_name2metrics)
    plt.show()


def main():
    evaluate_sgp()


if __name__ == '__main__':
    main()
