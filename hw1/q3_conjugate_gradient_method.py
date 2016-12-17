import matplotlib.pyplot as plt
import numpy

import blur


class ConjugateGradient(object):
    """
    Minimize 0.5 * xT * Q * x - bT * x.
    """
    def __init__(self, A, b, x0):
        assert b.ndim == 1
        assert A.shape == (len(b), len(b))
        self._A = A
        self._Q = A.T.dot(A)
        self._b_original = b
        self._b = A.T.dot(b)
        self._k = 0
        self._x = x0
        self._g = self._next_g(x0)
        self._d = -self._g
        self._beta = None
        self._alpha = None

    def _next_d(self, g, beta, d):
        return -g + beta * d

    def _next_g(self, x):
        return self._Q.dot(x) - self._b

    def _next_x(self, x, alpha, d):
        return x + alpha * d

    def _next_alpha(self, g, d):
        return - numpy.inner(g, d) / numpy.inner(d, self._Q.dot(d))

    def _next_beta(self, g, d):
        return numpy.inner(g, self._Q.dot(d)) / numpy.inner(d, self._Q.dot(d))

    def step(self):
        self._alpha = self._next_alpha(self._g, self._d)
        self._x = self._next_x(self._x, self._alpha, self._d)
        self._g = self._next_g(self._x)
        self._beta = self._next_beta(self._g, self._d)
        self._d = self._next_d(self._g, self._beta, self._d)
        self._k += 1

    def state(self):
        return self._x

    def gradient(self):
        return self._next_g(self._x)

    def value(self):
        return numpy.linalg.norm(self._A.dot(self._x) - self._b_original) ** 2


def main():
    n = 128
    A, b, real_x = blur.blur(n, 3, 0.8)
    x0 = numpy.zeros(b.shape)
    b_new = A.T.dot(b)
    search = ConjugateGradient(A, b_new, x0)
    num_iters = 100

    gradient_norms = []
    values = []
    for i in xrange(num_iters):
        gradient = search.gradient()
        gradient_norms.append(numpy.linalg.norm(gradient))
        values.append(search.value())
        search.step()
    x = search.state()
    plt.plot(values[2:], 'ro')
    plt.savefig('values_cgm.png')
    plt.cla()
    plt.plot(gradient_norms[2:], 'b+')
    plt.savefig('gradient_norms_cgm.png')
    blur.save_array_as_img(b, "converge_test_b_cgm")
    blur.save_array_as_img(x, "converge_test_found_x_cgm")
    blur.save_array_as_img(real_x, "converge_test_real_x_cgm")


if __name__ == '__main__':
    main()
