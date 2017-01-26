import numpy as np
import itertools
D = np.array([1, 2, 3, 2, 1], dtype=np.float)
A = []
for i in xrange(4):
    a = np.zeros(shape=(2, 5), dtype=np.float)
    a[0, i] = a[1, i + 1] = 1.
    A.append(a)


class PrimalState(object):
    def __init__(self, v, x):
        assert isinstance(v, np.ndarray)
        assert isinstance(x, np.ndarray)
        assert x.shape == D.shape
        assert v.shape == (4, 2)
        self._v = v.copy()
        self._x = x.copy()

    def full_score(self):
        return self.score()

    def score_primal(self):
        return self.score()

    def score(self):
        square_part = np.linalg.norm(self._x - D, ord=2) ** 2
        root_part = sum([np.inner(np.dot(A[i], self._x), self._v[i])
                         for i in xrange(4)])
        return square_part + root_part

    def mult_AiTvi(self):
        """
        sum(Ai_transpose * v-I)
        :return:
        """
        # This could be a done in a single un-readble line.
        res = np.zeros_like(self._x)
        for a, v in zip(A, self._v):
            res += np.dot(a.T, v)
        return res

    def gradient(self):
        x_grad = 2 * (self._x - D) + self.mult_AiTvi()
        assert x_grad.shape == self._x.shape
        v_grad = np.stack(tuple(np.dot(A[i], self._x) for i in xrange(4)))
        assert v_grad.shape == self._v.shape
        res = np.concatenate((x_grad.flatten(), v_grad.flatten()))
        return res

    def as_vec(self):
        return np.concatenate((self._x.flatten(), self._v.flatten()))

    @classmethod
    def random_state(cls):
        # Add 1 to make sure x2 >= 1
        return PrimalState(np.random.random(size=(4, 2)), D)

    @classmethod
    def from_vec(cls, vec):
        x = vec[:D.shape[0]]
        v = np.stack((vec[len(x) + i * 2: len(x) + i * 2 + 2] for i in xrange(4)))
        return PrimalState(v, x)

    def __str__(self):
        return "v:\n{}\nx:\n{}".format(self._v, self._x)

    def projection(self):
        actual_x = np.array([np.dot(A[i], self._x) for i in xrange(4)], dtype=np.float)
        v = project_to_l2_ball(actual_x)
        return PrimalState(v, self._x)


def iter_array(a):
    return itertools.product(*[range(s) for s in a.shape])


def test_gradient():
    for state_cls in (DualState, PrimalState):
        s = state_cls.random_state()
        x = s.as_vec()
        h = 1e-5
        gradient = s.gradient()
        fx = s.score()

        for i in iter_array(gradient):
            x[i] += h
            fx_plus_h = state_cls.from_vec(x).score()
            x[i] -= h
            g = (fx_plus_h - fx) / h
            assert abs(g - gradient[i]) < 0.0001


def test_from_to_vec():
    s = PrimalState.random_state()
    assert np.all(PrimalState.from_vec(s.as_vec()).as_vec() == s.as_vec())


class DualState(object):
    LIPSCHITZ_L = 2.

    def __init__(self, v):
        assert isinstance(v, np.ndarray)
        assert v.shape == (4, 2)
        self._v = v.copy()

    def _generate_x(self):
        return D - 0.5 * PrimalState(self._v, D).mult_AiTvi()

    def generate_primal(self):
        return PrimalState(self._v, self._generate_x())

    def full_score(self):
        return self.score()

    def score_primal(self):
        return self.generate_primal().projection().score()

    def score(self):
        return -self.generate_primal().score()

    def gradient(self):
        s = PrimalState(self._v, D).mult_AiTvi()
        square_part = - 1. / 2. * np.stack(tuple(np.dot(A[i], s) for i in xrange(4)))
        linear_part = np.stack(tuple(np.dot(D, A[i].T) for i in xrange(4)))
        v_grad = square_part + linear_part
        assert v_grad.shape == self._v.shape
        return -v_grad

    def as_vec(self):
        return self._v.copy()

    @classmethod
    def random_state(cls):
        # Add 1 to make sure x2 >= 1
        return DualState(np.random.random(size=(4, 2)))

    @classmethod
    def from_vec(cls, vec):
        return DualState(vec)

    def __str__(self):
        return "Dual:\n{}".format(self.generate_primal())

    def projection(self):
        return DualState(project_to_l2_ball(self._v))


def project_vector(v_i):
    # v_i = np.max((np.zeros_like(v_i), v_i), axis=0)
    norm = np.linalg.norm(v_i, ord=2)
    return v_i / norm
    # if norm <= 1:
    #     return v_i
    # else:
    #     return v_i / norm


def project_to_l2_ball(v):
    return np.stack(tuple(map(project_vector, v)))


test_from_to_vec()
test_gradient()
