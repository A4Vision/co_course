from hw2.random_problem import SearchState
from scipy import spatial


class SearchMethod(object):
    def __init__(self, search_state):
        assert isinstance(search_state, SearchState)
        self._state = search_state

    def step(self):
        raise NotImplementedError

    def euclidean_distance_to_target(self, true_x):
        return spatial.distance.euclidean(true_x, self._state)

    def deterministic_gradient_size(self):
        return spatial.distance.norm(self._state.middle_subgradient(), ord=2)



