import numpy as np
import unittest
from hw3.q1.utils import q1_dual_search_state


class TestDualGradient(unittest.TestCase):
    def test_gradient_primal(self):
        for j in xrange(20):
            state = q1_dual_search_state.Q1DualState.random_state()
            x = state.as_vec()
            epsilon = 1e-5
            h = np.zeros_like(x)
            fx = state.as_q1_state().full_score()
            gradient = state.gradient_primal()
            for i in xrange(2):
                print i
                h[i] = epsilon
                other = q1_dual_search_state.Q1DualState.from_vec(x + h)
                f_x_plus_h = other.as_q1_state().full_score()
                gradient_i = (f_x_plus_h - fx) / epsilon
                self.assertAlmostEqual(gradient_i, gradient[i], places=3)
                h[i] = 0

    def test_gradient_penalty(self):
        for j in xrange(20):
            state = q1_dual_search_state.Q1DualState.random_state()
            x = state.as_vec()
            epsilon = 1e-5
            h = np.zeros_like(x)
            fx = state.score_penalty()
            gradient = state.gradient_penalty()
            for i in xrange(2):
                print i
                h[i] = epsilon
                other = q1_dual_search_state.Q1DualState.from_vec(x + h)
                f_x_plus_h = other.score_penalty()
                gradient_i = (f_x_plus_h - fx) / epsilon
                self.assertAlmostEqual(gradient_i, gradient[i], places=3)
                h[i] = 0

