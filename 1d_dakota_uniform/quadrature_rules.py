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

    if dist._str() == 'Amalia windrose':
        # f(x)
        #   |                   *
        #   |   ***            * *      **
        #   | **   *          *   **  **  *     ***
        #   |*      *        *      **     *  **   *
        #   |        *      *               **
        #   |         *    *
        # --+----------****-----+------------------+--
        #  lo          A  B     C                  hi    (x)
        bnd = dist.range()
        a = bnd[0]  # left boundary
        b = bnd[1]  # right boundary
        # Make sure the A, B, C values are the same than those in distribution
        A = 110  # Left boundary of zero probability region
        B = 140  # Right boundary of zero probability region
        C = 225  # Location of max probability
        r = b-a  # original range
        r = r[0] # get rid of the list
        R = r - (B-A) # modified range
        x = np.linspace(a, R, n+1)
        dx = x[1]-x[0]  # I could also get dx from dx = R/n
        # Modify with offset
        # N = 1
        # temp = int(np.floor(N/2))
        # for i in range(-temp, temp+1):
        #     offset = i*dx/N
        #     x = np.linspace(a+offset, R+offset, n+1)
        #     x = x[:-1]+dx/2  # Take the midpoints of the bins
        #     # print i, x
        N = 5
        i = [-2, -1, 0, 1, 2]
        offset = 0*dx/N
        x = np.linspace(a+offset, R+offset, n+1)
        # This is probably good place for if statement between rectangle and dakota.
        x = x[:-1]+dx/2  # Take the midpoints of the bins


        # Modify x, to start from the max probability location
        x = (C+x)%r
        y = []
        for xi in x:
            if A<C:
                if xi > A and xi < C:
                    xi = (xi + B-A)%r
                y.append(xi)
            else:
                if xi > A:
                    xi = (xi + B-A)%r
                y.append(xi)
        x = np.array(y)

        # Get the weights associated with the points locations
        w = []
        for xi in x:
            w.append(dist._cdf(xi+dx/2.) - dist._cdf(xi-dx/2.))
        w = np.array(w).flatten()

    else:
        # Don't modify the range at all.
        bnd = dist.range()
        a = bnd[0]
        b = bnd[1]
        dx = (b-a)/n
        x = np.linspace(a+dx/2, b-dx/2, n) # Maybe modify this and then take the midpoints.

        # Get the weights associated with the points locations
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


