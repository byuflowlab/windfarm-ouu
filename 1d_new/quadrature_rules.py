import numpy as np
from scipy.interpolate import interp1d


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


def rectangle2(n, dist):
    bnd = dist.range()
    a = bnd[0]
    b = bnd[1]
    step = (b-a)/n
    x = np.linspace(a+step/2, b-step/2, n)
    w = np.ones(n)*step  # Integration weight, the dx interval
    f = dist.pdf(x)
    w *= f
    return [x], w


def rectangle(n, dist):
    # This integrates the pdf exactly
    bnd = dist.range()
    a = bnd[0]
    b = bnd[1]
    step = (b-a)/n
    x = np.linspace(a+step/2, b-step/2, n)
    A = 1.8
    B = 12.552983

    F = lambda x: -np.exp(-(x/B)**A)
    # print x
    # print x-step/2., x+step/2.
    # print dist.mom(x+step/2.), dist.mom(x-step/2.)
    # w = dist.mom(x+step/2.) - dist.mom(x-step/2.)  # mom is cdf
    w = F(x+step/2.) - F(x-step/2.)
    # print w
    print np.sum(w)
    # w = np.ones(n)*step  # Integration weight, the dx interval
    # f = dist.pdf(x)
    # w *= f
    return [x], w


def rectangle3(n, dist):
    # This integrates the pdf exactly
    bnd = dist.range()
    a = bnd[0]
    b = bnd[1]
    step = (b-a)/n
    x = np.linspace(a+step/2, b-step/2, n)
    w = []
    for xi in x:
        w.append(cdf(xi+step/2.) - cdf(xi-step/2.))
    w = np.array(w).flatten()
    print w
    print np.sum(w)
    print 'aloha'
    # w = np.ones(n)*step  # Integration weight, the dx interval
    # f = dist.pdf(x)
    # w *= f
    return [x], w



def _wind_rose_func():
    inputfile = 'amalia_windrose_8.txt'
    wind_data = np.loadtxt(inputfile)
    wind_data[wind_data == 0] = 2.00000000e-05  # Update were the wind data is zero to the next lowest value
    step = 360/len(wind_data)
    wind_data = np.append(wind_data, wind_data[0])  # Include the value at 360, which is the same as 0.
    wind_data = wind_data/step  # normalize for the [0, 360] range.
    x = np.array(range(0,360+1,step))
    f = interp1d(x, wind_data)
    return f

def cdf(x):
    # Integrate by rectangle rule
    dx = 0.001  # interval spacing
    f = _wind_rose_func()
    cdf = []
    # for x_i in x[0]:
    for x_i in x:
        X = np.arange(dx/2,x_i,dx)
        cdf.append(np.sum(f(X)*dx))  # integration by rectangle rule.
        # cdf.append(np.sum(X*f(X)*dx))  # I can use this to calculate the mean, when x is 360
    return np.array(cdf)