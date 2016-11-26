import numpy


class GradientDescentSearch(object):
    def __init__(self, A, b):
        self._A = A
        self._b = b
        self._ATA = A.T.dot(A)
        self._ATb = A.T.dot(b)

    def value(self, x):
        return numpy.linalg.norm(self._A.dot(x) - self._b) ** 2

    def gradient(self, x):
        return 2 * (self._ATA.dot(x) - self._ATb)

    def line_search(self, x, gradient):
        ATAg = self._ATA.dot(gradient)
        A = numpy.inner(gradient, ATAg)
        B = (-2 * numpy.inner(gradient, self._ATb) + 2 * numpy.inner(x, ATAg))
        return -B / (2 * A)

    def step(self, x, gradient):
        alpha = self.line_search(x, gradient)
        return x + alpha * gradient
