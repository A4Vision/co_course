from hw2 import random_problem


class SearchMethod(object):
    def __init__(self, search_state):
        self._state = search_state

    def step(self, eta):
        raise NotImplementedError

    def state(self):
        return self._state



