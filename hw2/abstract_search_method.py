from hw2 import random_problem


class SearchMethod(object):
    def __init__(self, search_state):
        assert isinstance(search_state, random_problem.SearchState)
        self._state = search_state

    def step(self):
        raise NotImplementedError

    def state(self):
        return self._state



