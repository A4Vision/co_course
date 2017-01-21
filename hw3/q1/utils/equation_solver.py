import numpy as np
EPSILON = 1e-4


def real_roots(polynomial_coefs, sign):
    """
    Returns the real (i.e. not complex) roots of a given polynomial,
    with a given sign (np.sign).
    :param polynomial_coefs:
    :param sign:
    :return:
    """
    roots = np.roots(polynomial_coefs)
    real = [x.real for x in roots if abs(x.imag) < EPSILON and np.sign(x.real) == sign]
    return real
