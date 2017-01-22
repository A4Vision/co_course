import numpy as np
import math

from hw3.q1.utils import projection

D1, D2, D3 = 1, 2, 3


class Q1State(object):
    # L=4 is the Lipschitz constant for the function:
    #   2 * (x_1 - 1) ** 2 + 2 * (x_2 - 2) ** 2 + (x_3 - 1) ** 2 + sqrt(y1) + sqrt(y2)
    #   in the domain x2 >= 1, y_1 >= x1 ** 2 + x2 ** 2, y2 >= x2 ** 2 + x3 ** 2
    LIPSCHITZ_L = 4

    def __init__(self, x1, x2, x3, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.y1 = y1
        self.y2 = y2

    def full_score(self):
        square_part = 2 * (self.x1 - D1) ** 2 + 2 * (self.x2 - D2) ** 2 + (self.x3 - D3) ** 2
        root_part = 2 * math.sqrt(self.y1) + 2 * math.sqrt(self.y2)
        return square_part + root_part

    def score_primal(self):
        return self.score()

    def score(self):
        y1 = self.x1 ** 2 + self.x2 ** 2
        y2 = self.x2 ** 2 + self.x3 ** 2
        full_state = Q1State(self.x1, self.x2, self.x3, y1, y2)
        return full_state.full_score()

    def gradient(self):
        assert self.y1 != 0 and self.y2 != 0
        return np.array([4 * (self.x1 - D1), 4 * (self.x2 - D2), 2 * (self.x3 - D3),
                         1. / math.sqrt(self.y1), 1. / math.sqrt(self.y2)], dtype=np.float64)

    def naive_gradient_3_variables(self):
        assert self.y1 != 0 and self.y2 != 0
        y1 = self.x1 ** 2 + self.x2 ** 2
        y2 = self.x2 ** 2 + self.x3 ** 2
        return np.array([4 * (self.x1 - D1) + 2 * self.x1 / math.sqrt(y1),
                         4 * (self.x2 - D2) + 2 * self.x2 * (1. / math.sqrt(y1) + 1. / math.sqrt(y2)),
                         2 * (self.x3 - D3) + 2 * self.x3 / math.sqrt(y2), 0, 0], dtype=np.float64)

    def as_vec(self):
        return np.array([self.x1, self.x2, self.x3, self.y1, self.y2], dtype=np.float64)

    @classmethod
    def random_state(cls):
        # Add 1 to make sure x2 >= 1
        vec = 1. + np.random.random(size=5)
        return Q1State(*vec.tolist())

    @classmethod
    def from_vec(cls, vec):
        return cls(*vec.tolist())

    def __str__(self):
        return "x1={} x2={} x3={} x4={} x5={}".format(self.x1, self.x2, self.x3, self.x2, self.x1)

    def projection(self):
        return Q1State(*projection.project_to_parabloids_intersection(*self.as_vec()))