"""
Projection to
    y_i >= (x_i ** 2 + x_(i+1) ** 2)
"""
from __future__ import division

import numpy as np

from hw3.q1.utils import equation_solver, q1_search_state


def project_to_parabloid_epigraph(x0, y0, z0):
    """
    Euclidean projection to the the set:
        {(x, y, z) | z ** 2 >= x ** 2 + y ** 2}
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


def project_to_parabloids_intersection(q1a_state):
    x1, x2, y1 = project_to_parabloid_epigraph(q1a_state.x1, q1a_state.x2, q1a_state.y1)
    x2, x3, y2 = project_to_parabloid_epigraph(x2, q1a_state.x3, q1a_state.y2)
    return q1_search_state.Q1State(x1, x2, x3, y1, y2)
