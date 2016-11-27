__author__ = 'Amit Botzer'

import numpy
import matplotlib.pyplot as plt
from q4_gradient_method import *
from q4_heavy_ball_method import *

def main():
    n = 50
    num_iters = 1000
    optimal_f_val = 0
    x0 = numpy.ones((n, 1))

    # alpha = 1, gamma = 0.2, x0 = ones(50,1)
    gm_search = GradientMethod(alpha=1, gamma=0.2, x0=x0)
    gm_differences = []
    gm_average_differences = []
    gm_sum = 0.0
    for i in xrange(num_iters):
        x_k = gm_search.state()
        gm_sum += x_k
        gm_average_differences.append(gm_search.f(gm_sum / (i+1)) - optimal_f_val)
        f_xk = gm_search.f(x_k)
        gm_differences.append(f_xk - optimal_f_val)
        gm_search.step()

    # alpha = 0.2, gamma = 0.2, x0 = ones(50,1), beta = 0.8
    hbm_search = HeavyBallMethod(alpha=0.2, gamma=0.2, x0=x0, beta=0.8)
    hbm_differences = []
    hbm_average_differences = []
    hbm_sum = 0.0
    for i in xrange(num_iters):
        x_k = hbm_search.state()
        hbm_sum += x_k
        hbm_average_differences.append(hbm_search.f(hbm_sum / (i+1)) - optimal_f_val)
        f_xk = hbm_search.f(x_k)
        hbm_differences.append(f_xk - optimal_f_val)
        hbm_search.step()

    x = numpy.arange(100)
    plt.figure(1)
    plt.plot(x, gm_differences[:100], 'g')
    plt.plot(x, hbm_differences[:100], 'r')
    plt.xlabel('number of iterations')
    plt.ylabel('f(x_k) - f(x*)')
    plt.legend(['GM', 'HBM'], loc='upper right')
    plt.savefig('gm_hbm_convergance.png')

    plt.figure(2)
    x = numpy.arange(1000)
    plt.plot(x, gm_average_differences, 'b')
    plt.plot(x, hbm_average_differences, 'r')
    plt.xlabel('number of iterations')
    plt.ylabel('f(x_N) - f(x*)')
    plt.legend(['GM', 'HBM'], loc='upper right')
    plt.savefig('average_convergance.png')
    plt.show()


if __name__ == '__main__':
    main()
