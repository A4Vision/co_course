import numpy as np


def get_real_root_with_sign(coefs, sign, max_results=1):
    roots = np.roots(coefs)
    real = [x.real for x in roots if abs(x.imag) < 1e-4 and np.sign(x.real) == sign]
    assert 1 <= len(real)  <= max_results
    if max_results == 1:
        return real[0]
    else:
        return real
