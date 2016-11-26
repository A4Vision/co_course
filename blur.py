import numpy as np
from scipy.linalg import toeplitz
from scipy import sparse

import PIL
import PIL.Image
import numpy

IMAGE = "squirrel.jpg"


def blur1(n, band, sigma):
    image = PIL.Image.open(IMAGE)
    resized_grayscale = image.resize((n, n), ).convert("L")
    x = numpy.asarray(resized_grayscale, dtype=numpy.float32).flatten()
    identity = numpy.identity(n ** 2, dtype=numpy.float32) * 4
    for i in xrange(n ** 2):
        for j in xrange(max(0, i - 2), min(n ** 2, i + 3)):
            if abs(i - j) == 1:
                identity[i, j] = 2
            if abs(i - j) == 2:
                identity[i, j] = 1
    A = identity / (1 + 2 + 4 + 2 + 1.)
    b = numpy.dot(A, x)
    b += numpy.random.normal(0., sigma, size=b.shape)
    return A, b, x


def save_array_as_img(x, title):
    print x.shape
    print numpy.max(x), numpy.min(x)
    assert x.ndim == 1
    n = int(x.shape[0] ** 0.5)
    assert x.shape[0] ** 0.5 == n
    shaped = x.reshape((n, n))
    # print numpy.float32(shaped) - numpy.float32(numpy.uint8(shaped))
    im = PIL.Image.fromarray(numpy.uint8(shaped))
    im.save(open(title + ".jpg", "wb"), format="jpeg")


def blur(N, band=3, sigma=0.7):
    z = np.concatenate([np.exp(-(np.r_[:band]**2)/(2*sigma**2)), np.zeros(N-band)])
    A = toeplitz(z)
    A = sparse.csr_matrix(A)
    A = (1/(2*np.pi*sigma**2))*sparse.kron(A,A)

    # Start with an image of all zeros.
    x = np.zeros((N,N))
    N2 = N/2
    N3= N/3
    N6 = N/6
    N12 = N/12

    # Add a large ellipse.
    ys,xs = np.ogrid[:2*N6,:2*N3]
    T = (((ys-N6)/N6)**2 + ((xs-N3)/N3)**2 < 1).astype(np.float64)
    x[:2*N6,N3:3*N3] = T

    # Add a smaller ellipse.
    T = (((ys-N6)/N6)**2 + ((xs-N3)/N3)**2 < 0.6).astype(np.float64)
    x[N6:3*N6,N3:3*N3] += 2*T
    # Correct for overlap.
    x[x>2]=2

    # Add a triangle.
    ys,xs = np.ogrid[:N3,:N3]
    T = ((xs+ys)<=N3-2).astype(np.float64)
    x[N3+N12:2*N3+N12,:N3] = 3*T

    # Add a cross.
    x[N2+N12:N2+N12+2*N6+1,N2+N6]=4
    x[N2+N12+N6,N2:N2+2*N6+1]=4

    # Make sure x is N-times-N, and stack the columns of x.
    x = x[:N,:N].ravel()

    _, x, _ = blur1(N, 1, 1)

    b = A*x
    return A, b, x


def main():
    _, b, x = blur(128, 3, 0.8)
    save_array_as_img(x, "original")
    save_array_as_img(b, "blurred")

if __name__ == '__main__':
    main()
