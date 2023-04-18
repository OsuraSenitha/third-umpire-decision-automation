import numpy as np
import perfplot
import math


def f(x):
    print(x)
    return hex(x)[2:].rjust(2, "0")


vf = np.vectorize(f)


def array_for(x):
    return np.array([f(xi) for xi in x])


def array_map(x):
    return np.array(list(map(f, x)))


def fromiter(x):
    return np.fromiter((f(xi) for xi in x), x.dtype)


def vectorize(x):
    return np.vectorize(f)(x)


def vectorize_without_init(x):
    return vf(x)

if __name__ == "__main__":
    b = perfplot.bench(
        setup=np.random.rand,
        n_range=[int(2 ** k) for k in range(20)],
        kernels=[
            f,
            array_for,
            array_map,
            fromiter,
            vectorize,
            vectorize_without_init,
        ],
        xlabel="len(x)",
    )
    b.save("out1.svg")
    b.show()