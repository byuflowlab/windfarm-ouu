import numpy as np


def trapezoid(n, dist):
    bnd = dist.range()
    a = bnd[0]
    b = bnd[1]
    x = np.linspace(a, b, n+1)
    w = np.ones(n+1)*(b-a)/(2*n)
    w[0] *= 0.5
    w[-1] *= 0.5
    f = dist.pdf(x)
    w *= f
    return [x], w


def rectangle(n, dist):
    bnd = dist.range()
    a = bnd[0]
    b = bnd[1]
    step = (b-a)/n
    x = np.linspace(a+step/2, b-step/2, n)
    w = np.ones(n)*step  # Integration weight, the dx interval
    f = dist.pdf(x)
    w *= f
    return [x], w
