import unittest
import numpy
import blur
import q3_gradient_descent
import matplotlib.pyplot as plt


class GradientDescentTest(unittest.TestCase):
    def testGradient(self):
        n = 50
        A, _, b = blur.blur(n, 3, 0.7)
        search = q3_gradient_descent.SearchSquare(A, b)
        x0 = numpy.random.normal(128, 50, size=n ** 2)
        h = numpy.zeros(shape=x0.shape)
        f_x0 = search.value(x0)
        gradient_x0 = search.gradient(x0)
        num_tests = 20
        epsilon = 0.01
        delta = numpy.linalg.norm(gradient_x0) / gradient_x0.size * 0.1
        for i in xrange(num_tests):
            print 'i', i
            coordinate = numpy.random.randint(0, len(x0))
            h[coordinate] += epsilon
            f_delta = search.value(x0 + h) - f_x0
            gradient_approx = f_delta / epsilon
            print gradient_approx, gradient_x0[coordinate]
            h[coordinate] -= epsilon
            self.assertAlmostEqual(gradient_approx, gradient_x0[coordinate], delta=delta)

    def testLineSearch(self):
        n = 50
        A, _, b = blur.blur(n, 3, 0.7)
        search = q3_gradient_descent.SearchSquare(A, b)
        for i in xrange(20):
            print 'i', i
            x0 = numpy.random.normal(128, 50, size=n ** 2)
            self.assertEqual(b.shape, x0.shape)
            gradient_x0 = search.gradient(x0)
            best_alpha = search.line_search(x0, gradient_x0)
            value_at_alpha = search.value(x0 + best_alpha * gradient_x0)
            print 'best_alpha', best_alpha
            for other_alpha in (best_alpha * 0.98, best_alpha * 1.02):
                print other_alpha / best_alpha
                value_at_other_alpha = search.value(x0 + other_alpha * gradient_x0)
                print value_at_other_alpha, value_at_alpha
                self.assertGreater(value_at_other_alpha, value_at_alpha)

    def testConverge(self):
        n = 50
        A, real_x, b = blur.blur(n, 3, 4)
        search = q3_gradient_descent.SearchSquare(A, b)
        # x0 = numpy.random.normal(128, 50, size=n ** 2)
        x0 = numpy.zeros(b.shape)
        num_iters = 100
        x = x0
        gradient_norms = []
        values = []
        for i in xrange(num_iters):
            gradient = search.gradient(x)
            gradient_norms.append(numpy.linalg.norm(gradient))
            values.append(search.value(x))
            x = search.step(x, gradient)
        gradient_x = search.gradient(x)
        gradient_size = numpy.linalg.norm(gradient_x)
        epsilon = 10
        self.assertGreater(epsilon, gradient_size)

        plt.plot(values[2:], 'ro')
        plt.savefig('values.png')
        plt.cla()
        plt.plot(gradient_norms[2:], 'b+')
        plt.savefig('gradient_norms.png')
        blur.save_array_as_img(b, "converge_test_b")
        blur.save_array_as_img(x, "converge_test_found_x")
        blur.save_array_as_img(real_x, "converge_test_real_x")
