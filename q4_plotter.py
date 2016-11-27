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
    gm_averages = []
    gm_sum = 0.0
    for i in xrange(num_iters):
        x_k = gm_search.state()
        gm_sum += x_k
        gm_averages.append(gm_sum / (i+1))
        f_xk = gm_search.f(x_k)
        gm_differences.append(f_xk - optimal_f_val)
        gm_search.step()

    # alpha = 0.2, gamma = 0.2, x0 = ones(50,1), beta = 0.8
    hbm_search = HeavyBallMethod(alpha=0.2, gamma=0.2, x0=x0, beta=0.8)
    hbm_differences = []
    hbm_averages = []
    hbm_sum = 0.0
    for i in xrange(num_iters):
        x_k = hbm_search.state()
        hbm_sum += x_k
        hbm_averages.append(hbm_sum / (i+1))
        f_xk = hbm_search.f(x_k)
        hbm_differences.append(f_xk - optimal_f_val)
        hbm_search.step()

    x = numpy.arange(100)
    plt.plot(x, gm_differences[:100], 'g')
    plt.plot(x, hbm_differences[:100], 'r')
    plt.legend(['GM', 'HBM'], loc='upper right')
    plt.show()

    #plt.savefig('values_cgm.png')
    #plt.cla()
    #plt.plot(gradient_norms[2:], 'b+')
    #plt.savefig('gradient_norms_cgm.png')
    #blur.save_array_as_img(b, "converge_test_b_cgm")
    #blur.save_array_as_img(x, "converge_test_found_x_cgm")
    #blur.save_array_as_img(real_x, "converge_test_real_x_cgm")

if __name__ == '__main__':
    main()
