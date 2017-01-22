import numpy as np
from hw3.q1.utils import projection
from hw3.q1.utils import q1_search_state
from hw3.q1.utils.q1_search_state import D1, D2, D3


class Q1DualState(object):
    LIPSCHITZ_L = 425.

    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2

    def as_q1_state(self):
        x1 = 2 / (2. + self.l1)
        x2 = 4 / (2. + self.l1 + self.l2)
        x3 = 3 / (1. + self.l2)
        y1 = 1 / self.l1 ** 2
        y2 = 1 / self.l2 ** 2
        return q1_search_state.Q1State(x1, x2, x3, y1, y2)

    def score_primal(self):
        return self.as_q1_state().score()

    def score_penalty(self):
        primal = self.as_q1_state()
        return (self.l1 * (primal.x1 ** 2 + primal.x2 ** 2 - primal.y1) +
                self.l2 * (primal.x2 ** 2 + primal.x3 ** 2 - primal.y2))

    def score(self):
        return self.score_primal() + self.score_penalty()

    def gradient_penalty(self):
        primal = self.as_q1_state()
        a = primal.x1 ** 2 + primal.x2 ** 2 + primal.y1
        common = -0.5 * primal.x2 ** 3
        b = self.l1 * (-primal.x1 ** 3 + common)
        c = self.l2 * common
        val1 = a + b + c
        a = self.l1 * common + primal.x3 ** 2 + primal.x2 ** 2
        b = self.l2 * (-18 / 27. * primal.x3 ** 3 + common)
        c = primal.y2
        val2 = a + b + c
        return np.array([val1, val2], dtype=np.float64)

    def gradient_primal(self):
        diff1 = 4 * (2 / (2. + self.l1) - D1) * (-2 / (2. + self.l1) ** 2)
        diff2 = 4 * (4 / (2. + self.l1 + self.l2) - D2) * (-4 / (2. + self.l1 + self.l2) ** 2)
        diff3 = 2 * (3 / (1. + self.l2) - D3) * (-3 / (1. + self.l2) ** 2)
        val1 = diff1 + diff2 - 2. / (self.l1 ** 2)
        val2 = diff2 + diff3 - 2. / (self.l2 ** 2)
        return np.array([val1, val2], dtype=np.float64)

    def gradient(self):
        return self.gradient_penalty() + self.gradient_primal()

    def as_vec(self):
        return np.array([self.l1, self.l2], dtype=np.float64)

    @classmethod
    def random_state(cls):
        # Add 1 to make sure x2 >= 1
        vec = 1. / 5 ** 0.5 + np.random.random(size=2)
        return Q1DualState(*vec.tolist())

    @classmethod
    def from_vec(cls, vec):
        return cls(*vec.tolist())

    def __str__(self):
        return "l1={} l2={}".format(self.l1, self.l2)

    def projection(self):
        return Q1DualState(*projection.project_dual(self.l1, self.l2))
