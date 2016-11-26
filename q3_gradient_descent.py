import numpy


class SearchSquare(object):
    def __init__(self, A, b):
        self._A = A
        self._b = b
        self._ATA = numpy.dot(A.T, A)
        self._ATb = numpy.dot(A.T, b)

    def value(self, x):
        return numpy.linalg.norm(numpy.dot(self._A, x) - self._b) ** 2

    def gradient(self, x):
        return 2 * (numpy.dot(self._ATA, x) - self._ATb)

    def line_search(self, x, gradient):
        ATAg = numpy.dot(gradient, self._ATA)
        A = numpy.inner(gradient, ATAg)
        B = (-2 * numpy.inner(gradient, self._ATb) + 2 * numpy.inner(x, ATAg))
        return -B / (2 * A)

    def step(self, x, gradient):
        alpha = self.line_search(x, gradient)
        return x + alpha * gradient
