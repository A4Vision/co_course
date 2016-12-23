from matplotlib import pyplot as plt
import numpy as np
from hw2 import random_problem
from hw2 import subgradient_projection_method
from scipy.spatial import distance
COLORS = ["red", "green", "blue", "black"]
N_steps = 120
FIRST_STEP_TO_IGNORE_IN_PLOT = 100


def evaluate_sgp():
    problem, x_true = random_problem.randomize_problem()
    f, (scores_ax, gradients_ax, distances_ax, deltas_ax) = plt.subplots(4, sharex=True)
    f.suptitle("Subgradient Projection Method")
    scores_ax.set_title("Score - ||Ax - b||")
    gradients_ax.set_title("Gradient norm - ||gradient(x)||")
    distances_ax.set_title("Distance to real - ||x - x_true||")
    deltas_ax.set_title("Step delta size - ||x - x_previous||")
    step_size_selectors = [subgradient_projection_method.DynamicStepSize(),
                          subgradient_projection_method.ConstantStepSize(N_steps),
                          subgradient_projection_method.OptimalStepKnownTargetValue(0),
                          subgradient_projection_method.SmallerThanOtherSelector(
                          subgradient_projection_method.OptimalStepKnownTargetValue(0), 0.5)]
    for step_selector, color in zip(step_size_selectors, COLORS):
        search_state = random_problem.SearchState(problem, np.ones_like(x_true) / len(x_true))
        searcher = subgradient_projection_method.SubgradientProjectionMethod(search_state, step_selector)
        scores = []
        gradients_sizes = []
        distances_to_x_true = []
        x_delta_size = []
        prev_x = search_state.x()
        for i in xrange(N_steps):
            searcher.step()
            state = searcher.state()
            scores.append(state.score())
            gradients_sizes.append(state.deterministic_gradient_size())
            distances_to_x_true.append(state.euclidean_distance_to_target(x_true))
            x_delta_size.append(distance.euclidean(state.x(), prev_x))
            prev_x = state.x()

        label = step_selector.__class__.__name__
        scores_ax.plot(scores[FIRST_STEP_TO_IGNORE_IN_PLOT:], label=label, color=color)
        scores_ax.legend()
        gradients_ax.plot(gradients_sizes[FIRST_STEP_TO_IGNORE_IN_PLOT:], label=label, color=color)
        gradients_ax.legend()
        distances_ax.plot(distances_to_x_true[FIRST_STEP_TO_IGNORE_IN_PLOT:], label=label, color=color)
        distances_ax.legend()
        deltas_ax.plot(x_delta_size[FIRST_STEP_TO_IGNORE_IN_PLOT:], label=label, color=color)
        deltas_ax.legend()
        plt.draw()

    plt.show()


def main():
    evaluate_sgp()


if __name__ == '__main__':
    main()