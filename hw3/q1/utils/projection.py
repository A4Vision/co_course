"""
Projection to
    y_i >= (x_i ** 2 + x_(i+1) ** 2)
"""
from __future__ import division

import numpy as np

from hw3.q1.utils import equation_solver


def project_to_parabloid_epigraph(x0, y0, z0):
    """
    Euclidean projection to the the set:
        {(x, y, z) | z >= x ** 2 + y ** 2}
    :param x0:
    :param y0:
    :param z0:
    :return:
    """
    if z0 >= x0 ** 2 + y0 ** 2:
        return x0, y0, z0
    if y0 == 0:
        y, x, z = project_to_parabloid_epigraph(y0, x0, z0)
        return x, y, z
    elif x0 == 0:
        coefs = [4 * y0, 0, 2 - 4 * z0, -2 * y0]
        ys = equation_solver.real_roots(coefs, np.sign(y0))
        assert len(ys) == 1
        y = ys[0]
        x = 0
        z = y ** 2 + x ** 2
        return x, y, z
    else:
        B = (y0 / x0) ** 2
        coefs = [2 * (B + 1),  0, 1. - 2 * z0, -x0]
        xs = equation_solver.real_roots(coefs, np.sign(x0))
        assert len(xs) == 1
        x = xs[0]
        y = y0 / x0 * x
        z = x ** 2 + y ** 2
        return x, y, z


def project_to_ball_hypergraph(x, z):
    """
    Euclidean projection to the the set:
        {(x, z) | z >= x ** 2 + 1}
    :param x:
    :param y:
    :param z:
    :return:
    """
    coefs = [2, 0, 1 - 2 * z, -x]
    xs = equation_solver.real_roots(coefs, np.sign(x))
    assert len(xs) == 1
    x = xs[0]
    z = x ** 2 + 1
    return x, z


def project_to_parabloid_and_y_greater_than1(x, y, z):
    x, y, z = project_to_parabloid_epigraph(x, y, z)
    if y < 1:
        y = 1.
        x, z = project_to_ball_hypergraph(x, z)
    return x, y, z


def project_to_parabloids_intersection(x1, x2, x3, y1, y2):
    x1, x2, y1 = project_to_parabloid_and_y_greater_than1(x1, x2, y1)
    x3, x2, y2 = project_to_parabloid_and_y_greater_than1(x3, x2, y2)
    return x1, x2, x3, y1, y2


def project_dual(l1, l2):
    return max(l1, 1 / 5. ** 0.5), max(l2, 1 / 13. ** 0.5)
