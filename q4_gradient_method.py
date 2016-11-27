import numpy


class GradientMethod(object):

    """
    Minimize f(x) where:
     f(x) = gamma * ||x|| - gamma ^ 2 / 2    if   ||x|| >= gamma
     f(x) = 1/2 * ||x|| ^ 2                  if   ||x|| <= gamma
    """

    def __init__(self, alpha, gamma, x0):
        self._x = x0
        self._gamma = gamma
        self._alpha = alpha
        self._g = self.gradient(x0)

    """
     f'(x) = gamma * x / ||x||   if   ||x|| >= gamma
     f'(x) = x                   if   ||x|| <= gamma
    """
    def gradient(self, x):
        x_norm = numpy.linalg.norm(x)
        if x_norm >= self._gamma:
            return self._gamma * x / x_norm
        else:
            return x

    def step(self):
        self._x = self._x - self._alpha * self._g
        self._g = self.gradient(self._x)

    def state(self):
        return self._x

    def f(self, x):
        x_norm = numpy.linalg.norm(x)
        if x_norm >= self._gamma:
            return self._gamma * x_norm - self._gamma ** 2 / 2
        else:
            return 1 / 2 * x_norm ** 2