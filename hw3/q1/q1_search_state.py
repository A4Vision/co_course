import numpy as np
import math
D1, D2, D3 = 1, 2, 3


class Q1State(object):
    __slots__ = ('x1', 'x2', 'x3', 'y1', 'y2',)

    def __init__(self, x1, x2, x3, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.y1 = y1
        self.y2 = y2

    def score(self):
        square_part = 2 * (self.x1 - D1) ** 2 + 2 * (self.x2 - D2) ** 2 + (self.x3 - D3) ** 2
        y1 = self.x1 ** 2 + self.x2 ** 2
        y2 = self.x2 ** 2 + self.x3 ** 2
        root_part = 2 * math.sqrt(y1) + 2 * math.sqrt(y2)
        return square_part + root_part

    def subgradient_vec(self):
        assert self.y1 != 0 and self.y2 != 0
        return np.array([4 * (self.x1 - D1), 4 * (self.x2 - D2), 2 * (self.x3 - D3),
                         1. / math.sqrt(self.y1), 1. / math.sqrt(self.y2)], dtype=np.float64)

    def subgradient_naive(self):
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
        vec = np.random.random(size=5)
        return Q1State(*vec.tolist())

    @classmethod
    def from_vec(cls, vec):
        return cls(*vec.tolist())
