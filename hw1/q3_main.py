import glob
import os
import sys

import numpy

try:
    import matplotlib.pyplot as plt
    pass
except:
    pass
import blur
import q3_gradient_descent
import q3_conjugate_gradient_method

OUTPUT_FILES = True


def solve_x(search, output_folder, title):
    if OUTPUT_FILES:
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
    print
    print title
    print "==========="
    NUM_ITERS = 100
    gradient_norms = []
    values = []
    images = []
    texts = []
    for i in xrange(NUM_ITERS):
        gradient_norms.append(numpy.linalg.norm(search.gradient()))
        values.append(search.value())
        if i in range(9, NUM_ITERS, 10):
            print 'iteration=', i + 1, 'value=', search.value()
            image_path = "{}/state{:02d}".format(output_folder, i)
            images.append(image_path)
            texts.append("Iteration {}".format(i + 1))
            if OUTPUT_FILES:
                blur.save_array_as_img(search.state(), image_path)
        search.step()

    if OUTPUT_FILES:
        plt.plot(range(9, len(values)), values[9:], 'ro--', label='||Ax - b|| ** 2')
        plt.title(title)
        plt.xlabel("iteration")
        plt.ylabel("Value - ||Ax - b|| ** 2")
        plt.legend()
        plt.savefig('{}/func_values.png'.format(output_folder))
        plt.cla()  # reset the plot

        # Plot the gradients norms...
        plt.plot(gradient_norms[2:], 'bo--', label="||gradient||")
        plt.title("Gradient Size")
        plt.legend()
        plt.savefig('{}/gradient_norms.png'.format(output_folder))
        plt.cla()

        with open("{}.html".format(output_folder), "wb") as f:
            images = glob.glob("q3_gd_output/state*.bmp")
            f.write(create_table(images, texts, 3, 4))


def solve_gradient_descent(N, A, b):
    algorithm = q3_gradient_descent.GradientDescent(A, b)
    x0 = numpy.zeros(b.shape)
    search = q3_gradient_descent.GradientDescentSearch(algorithm, x0)
    solve_x(search, "q3_gd_output", "Gradient Descent Values")


def solve_conjugate_gradient_method(N, A, b):
    x0 = numpy.zeros(b.shape)
    search = q3_conjugate_gradient_method.ConjugateGradient(A, b, x0)
    solve_x(search, "q3_cgm_output", "Conjugate Gradient Values")


def create_table(paths_list, texts_list, n_rows, n_columns):
    """
    Create HTML table with a table that contains the given files.
    :param paths_list: list of paths of images.
    :return:
    """
    assert n_rows * n_columns >= len(paths_list)
    table = [[''] * n_columns for _ in xrange(n_rows)]

    for i in xrange(n_rows):
        for j in xrange(n_columns):
            if i * n_columns + j < min(len(paths_list), len(texts_list)):
                index = i * n_columns + j
                path, text = paths_list[index], texts_list[index]
                img_html = '<img src="{}">'.format(path)
                text_html = "<p>{}</p>".format(text)
                table[i][j] = img_html + text_html

    return htmlify_table(table)


def htmlify_table(table):
    n_rows = len(table)
    n_columns = len(table[0])
    res = ''
    for i in xrange(n_rows):
        res += '<tr>'
        for j in xrange(n_columns):
            res += '<td>{}</td>'.format(table[i][j])
        res += '</tr>'
    return '<table>{}</table>'.format(res)


def main():
    global OUTPUT_FILES
    use_squirrel = ("squirrel" in sys.argv)
    use_squirrel = True
    # Don't allow output if could not import matplotlib
    OUTPUT_FILES = ("output" in sys.argv) and ('plt' in globals())
    OUTPUT_FILES = True
    N = 128
    A, b, real_x = blur.blur(N, 3, 0.8, use_squirrel)
    if OUTPUT_FILES:
        blur.save_array_as_img(b, "blurred")
        blur.save_array_as_img(real_x, "real")
    solve_gradient_descent(use_squirrel, A, b)
    solve_conjugate_gradient_method(use_squirrel, A, b)


if __name__ == '__main__':
    main()
