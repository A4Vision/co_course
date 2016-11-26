import PIL
import PIL.Image
import numpy

IMAGE = "squirrel.jpg"


def blur(n, band, sigma):
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
    return A, x, b


def save_array_as_img(x, title):
    assert x.ndim == 1
    n = int(x.shape[0] ** 0.5)
    assert x.shape[0] ** 0.5 == n
    shaped = x.reshape((n, n))
    print numpy.float32(shaped) - numpy.float32(numpy.uint8(shaped))
    im = PIL.Image.fromarray(numpy.uint8(shaped))
    im.save(open(title + ".png", "wb"), format="png")


def main():
    A, x, b = blur(150, 0, 0)
    save_array_as_img(x, "original")
    save_array_as_img(b, "blurred")

if __name__ == '__main__':
    main()
