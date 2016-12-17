import numpy as np


def project_into_simplex(x):
    """
    Projects the given point in R^n, to a point in the simplex:
        SUM(x_i) <= 1
    :param x: numpy array of dimension 1. Point to project.
    :return:
    """
    assert np.all(x >= 0)
    assert x.ndim == 1
    n = len(x)
    x_sorted = np.sort(x)[::-1]
    sums = np.cumsum(x_sorted) - 1
    averages = sums / np.arange(1, n + 1)
    slacks = x_sorted - averages
    rho = n - np.searchsorted(slacks[::-1], 0)
    theta = averages[rho - 1]

    return np.max([x - theta, np.zeros_like(x)], axis=0)


def project_into_l1_ball(x):
    if np.sum(np.abs(x)) <= 1:
        return x
    sgns = np.sign(x)
    projected = project_into_simplex(sgns * x)
    return sgns * projected


