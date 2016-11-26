import numpy
import matplotlib.pyplot as plt
import blur


class ConjugateGradient(object):
    """
    Minimize 0.5 * xT * Q * x - bT * x.
    """
    def __init__(self, Q, b, x0):
        assert b.ndim == 1
        assert Q.shape == (len(b), len(b))
        self._Q = Q
        self._b = b
        self._k = 0
        self._x = x0
        self._g = self._next_g(x0)
        self._d = -self._g
        self._beta = None
        self._alpha = None

    def _next_d(self, g, beta, d):
        return -g + beta * d

    def _next_g(self, x):
        return numpy.dot(self._Q, x) - self._b

    def _next_x(self, x, alpha, d):
        return x + alpha * d

    def _next_alpha(self, g, d):
        return - numpy.inner(g, d) / numpy.inner(d, numpy.dot(self._Q, d))

    def _next_beta(self, g, d):
        return numpy.inner(g, numpy.dot(self._Q, d)) / numpy.inner(d, numpy.dot(self._Q, d))

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
        return numpy.inner(self._x, numpy.dot(self._Q, self._x)) - 2 * numpy.inner(self._b, self._x)


def main():
    n = 50
    A, real_x, b = blur.blur(n, 3, 4)
    x0 = numpy.random.normal(128, 50, size=n ** 2)
    x0 = numpy.zeros(b.shape)
    Q = numpy.dot(A.T, A)
    b_new = numpy.dot(A.T, b)
    search = ConjugateGradient(Q, b_new, x0)
    num_iters = 40

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
