import numpy as np
import collections
from scipy.spatial import distance

Problem = collections.namedtuple("Problem", ("A", "b"))


def randomize_problem(m=80, n=100, seed=318):
    np.random.seed(seed)
    A = np.random.randn(m, n)
    x_true = np.random.random(size=n)
    x_true /= np.sum(x_true)  # Normalize x.
    b = np.dot(A, x_true)
    return Problem(A, b), x_true


class SearchState(object):
    def __init__(self, problem, x0):
        self._x = x0
        self._problem = problem

    def x(self):
        return np.array(self._x)

    def problem(self):
        return self._problem

    def A(self):
        return self._problem.A

    def b(self):
        return self._problem.b

    def score(self):
        return distance.norm(np.dot(self.A(), self.x()) - self.b(), ord=1)

    def move_to_x(self, x):
        return SearchState(self._problem, x)

    def random_subgradient(self):
        """
        Random subgradient of the target function.
        :return: SUM_i(ai_T * sign(ai_T * x - b_i)  ; ai_T * x - b_i != 0) +
                 SUM_i(ai_T * random(-1, 1)         ; ai_T * x - b_i == 0)
        """
        # Calculate SUM_i(ai_T * sign(ai_T * x - b_i)  ; ai_T * x - b_i != 0)
        diff = np.dot(self.A(), self.x()) - self.b()
        diff_signs = np.sign(diff)
        nonzeros_part = np.dot(diff_signs, self.A())
        # Calculate SUM_i(ai_T * random(-1, 1)         ; ai_T * x - b_i == 0)
        random_weights = np.random.uniform(-1, 1, size=self.A().shape[0])
        zeros = (diff_signs == 0)
        weighted_zeros = zeros * random_weights
        zeros_part = np.dot(weighted_zeros, self.A())

        return nonzeros_part + zeros_part

    def middle_subgradient(self):
        """
        Deterministic subgradient of the target function, where .
        :return: SUM_i(ai_T * sign(ai_T * x - b_i)  ; ai_T * x - b_i != 0)
        """
        diff = np.dot(self.A(), self.x()) - self.b()
        diff_signs = np.sign(diff)
        return np.dot(diff_signs, self.A())

    def euclidean_distance_to_target(self, x_true):
        return distance.euclidean(x_true, self.x())

    def deterministic_gradient_size(self):
        return distance.norm(self.middle_subgradient(), ord=2)
