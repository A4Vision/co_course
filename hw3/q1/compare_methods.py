import numpy as np
from matplotlib import pyplot as plt
from hw3.q1 import q1_search_state
from hw3.q1 import step_size
from hw3.q1 import subgradient_projection_method


def run_sgp():
    step_size_selector = step_size.DynamicStepSize()
    np.random.seed(123)
    state0 = q1_search_state.Q1State.random_state()
    method0 = subgradient_projection_method.SubgradientProjectionMethod(state0, step_size_selector,
                                                                        False)
    method1 = subgradient_projection_method.SubgradientProjectionMethod(state0, step_size_selector,
                                                                        True)
    scores0 = []
    scores1 = []
    for i in xrange(1000):
        method0.step(0.65)
        method1.step(0.2)
        scores0.append(method0.state().score())
        scores1.append(method1.state().score())
    plt.figure(figsize=(200, 400))
    plt.plot(scores0, 'r-*')
    plt.plot(scores1, 'b--')
    print zip(scores0, scores1)[800:]
    plt.show()
    print method0.state().as_vec()
    print method1.state().as_vec()




def main():
    run_sgp()


if __name__ == '__main__':
    main()