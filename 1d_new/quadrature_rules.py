import numpy as np


def rectangle(n, dist):
    """Modified rectangle. Integrates the pdf exactly

    Args:
        n: the number of bins
        dist: a chaospy distribution

    Returns:
        x: quadrature points
        w: quadrature weights
    """
    bnd = dist.range()
    a = bnd[0]
    b = bnd[1]
    dx = (b-a)/n
    x = np.linspace(a+dx/2, b-dx/2, n)
    w = []
    for xi in x:
        w.append(dist._cdf(xi+dx/2.) - dist._cdf(xi-dx/2.))
    w = np.array(w).flatten()
    return [x], w


def rectangle_unmodified(n, dist):
    """Applies the pure rectangle method

    Args:
        n: the number of bins
        dist: a chaospy distribution

    Returns:
        x: quadrature points
        w: quadrature weights
    """
    bnd = dist.range()
    a = bnd[0]
    b = bnd[1]
    dx = (b-a)/n
    x = np.linspace(a+dx/2, b-dx/2, n)
    w = np.ones(n)*dx  # Integration weight, the dx interval
    f = dist.pdf(x)
    w *= f
    return [x], w

