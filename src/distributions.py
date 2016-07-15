import chaospy as cp
import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import special


class amaliaWindRose(object):
    """The smoothed amalia distribution."""

    def __init__(self):

        # f(x)
        #   |                   *
        #   |   ***            * *      **
        #   | **   *          *   **  **  *     ***
        #   |*      *        *      **     *  **   *
        #   |        *      *               **
        #   |         *    *
        # --+----------****-----+------------------+--
        #  lo          A  B     C                  hi    (x)

        self.lo = 0.0
        self.hi = 360.0
        self.A = 110  # Left boundary of zero probability region
        self.B = 140  # Right boundary of zero probability region
        # self.C = 225  # Location of max probability

    def pdf(self, x):
        x = x.flatten()  # In the constructor of the distribution it gets made a 2d array for some reason. But not for cdf
        f = self._wind_rose_poly_func()  # This will give me different results for the rectangle method. It's because the values were different
        return f(x)

    def cdf(self, x):
        # Integrate by rectangle rule
        # dx = 0.001  # interval spacing
        cdf = []
        for x_i in np.array(x, copy=False, ndmin=1):  # makes it work if x is a scalar
            dx = x_i/100.  # interval spacing
            if x_i == 0:
                cdf.append(0.0)
            else:
                X = np.arange(dx/2, x_i, dx)
                cdf.append(np.sum(self.pdf(X)*dx))  # integration by rectangle rule
        return np.array(cdf)

    def str(self):
        return "Amalia windrose"

    def bnd(self):
        return (self.lo, self.hi)

    def _wind_rose_poly_func(self):
        def f(x):
            return np.array([self._f_helper(z) for z in x])
        return f

    def _windrose_polyfit(self, x):
        y = 493597.250387841  *np.power(x, 12) + \
            -207774.160030495 *np.power(x, 11) + \
            -413203.013010848 *np.power(x, 10) + \
            158080.893880027  *np.power(x, 9) + \
            127607.500730722  *np.power(x, 8) + \
            -44242.1722820275 *np.power(x, 7) + \
            -17735.2623897828 *np.power(x, 6) + \
            5422.11156037294  *np.power(x, 5) + \
            1057.31910521884  *np.power(x, 4) + \
            -253.807324825523 *np.power(x, 3) + \
            -19.8973363502958 *np.power(x, 2) + \
            1.43458543839655  *np.power(x, 1) + \
            1.05778787373732  *np.power(x, 0)
        return y

    def _f_helper(self, x):
        # Linear transformation from interval [a,b] to [-0.5,0.5]
        A = self.A
        B = self.B
        a = B  # 140
        b = self.hi - self.lo + A  # 470
        if x >= B:
            x1 = (x - (b+a)/2.) / (b-a)
            return self._windrose_polyfit(x1)/330  # The 330 comes from 360-(B-A)=R in quadrature rules
            # return self._windrose_polyfit(x1)/360
        elif x <= A:
            x1 = (x + 360 - (b+a)/2.) / (b-a)
            return self._windrose_polyfit(x1)/330
            # return self._windrose_polyfit(x1)/360
        else:
            # If I'm only calling the pdf I should not go in here, but when calling the cdf I get in here.
            return 0.0


class amaliaWindRoseRaw(object):
    """The raw amalia distribution."""

    def __init__(self):
        self.lo = 0.0
        self.hi = 360.0
        self.inputfile = '../WindRoses/amalia_windrose_8.txt'

    def _wind_rose_func(self):
        wind_data = np.loadtxt(self.inputfile)
        wind_data[wind_data == 0] = 2.00000000e-05  # Update were the wind data is zero to the next lowest value
        step = 360/len(wind_data)
        wind_data = np.append(wind_data, wind_data[0])  # Include the value at 360, which is the same as 0.
        wind_data = wind_data/step  # normalize for the [0, 360] range.
        x = np.array(range(0,360+1,step))
        f = interp1d(x, wind_data)
        return f

    def pdf(self, x):
        x = x.flatten()  # In the constructor of the distribution it gets made a 2d array for some reason. For this amalia class this flattening is unnecesary
        f = self._wind_rose_func()
        return f(x)

    def cdf(self, x):
        # Integrate by rectangle rule
        # dx = 0.001  # interval spacing
        cdf = []
        for x_i in np.array(x, copy=False, ndmin=1):  # makes it work if x is a scalar
            dx = x_i/100.  # interval spacing
            if x_i == 0:
                cdf.append(0.0)
            else:
                X = np.arange(dx/2, x_i, dx)
                cdf.append(np.sum(self.pdf(X)*dx))  # integration by rectangle rule
        return np.array(cdf)

    def str(self):
        return "Amalia windrose raw"

    def bnd(self):
        return (self.lo, self.hi)


class amaliaWindRoseRaw01(object):
    """The raw amalia distribution."""

    def __init__(self):
        self.lo = 0.0
        self.hi = 1.0
        self.inputfile = '../WindRoses/windrose_amalia_8ms.txt'

    def _wind_rose_func(self):
        wind_data = np.loadtxt(self.inputfile)
        # direction = wind_data[:, 0]
        # speed = wind_data[:, 1]
        probability = wind_data[:, 2]
        N = len(probability)
        probability[probability == 0] = 2e-5  # think about this, # Update were the wind data is zero to the next lowest value
        probability = np.append(probability, probability[0])  # Include the value at 360, which is the same as 0.
        probability = probability*N  # normalize for the [0, 1] range.
        x = np.linspace(0, 1, N+1)
        f = interp1d(x, probability)
        return f

    def pdf(self, x):
        x = x.flatten()  # In the constructor of the distribution it gets made a 2d array for some reason. For this amalia class this flattening is unnecesary
        f = self._wind_rose_func()
        return f(x)

    def cdf(self, x):
        # Integrate by rectangle rule
        # dx = 0.001  # interval spacing
        cdf = []
        for x_i in np.array(x, copy=False, ndmin=1):  # makes it work if x is a scalar
            dx = x_i/100.  # interval spacing
            if x_i == 0:
                cdf.append(0.0)
            else:
                X = np.arange(dx/2, x_i, dx)
                cdf.append(np.sum(self.pdf(X)*dx))  # integration by rectangle rule
        return np.array(cdf)

    def str(self):
        return "Amalia windrose raw [0,1]"

    def bnd(self):
        return (self.lo, self.hi)


class myWeibull(object):
    def __init__(self):
        self.a = 1.8
        self.b = 12.552983
        self.lo = 0.0
        self.hi = 80.0  # cdf(80) = 1, to machine precision

    def cdf(self, x):
        a = self.a
        b = self.b
        F = 1-np.exp(-(x/b)**a)
        return F

    def pdf(self, x):
        a = self.a
        b = self.b
        f = a/b * (x/b)**(a-1) * np.exp(-(x/b)**a)
        return f

    def mom(self, k):
        # I don't think providing the moments here changes anything down the road when
        # doing PC expansions. Although if I don't pass mom the moments calculated are
        # bad. It appears that they just take the moments assuming a uniform with
        # a range, as that obtained from the bounds.
        # Actually it does affect the PC expansions, this comes into play when getting
        # the orthogonal polynomials.
        a = self.a
        b = self.b
        return b * special.gamma(1.+k*1./a)

    def str(self):
        return "weibull(%s, %s)" % (self.a, self.b)

    def bnd(self):
        return (self.lo, self.hi)


class TruncatedWeibull(object):
    def __init__(self):
        self.a = 1.8
        self.b = 12.552983
        self.lo = 0.0
        self.hi = 30.0
        self.k = self.set_truncation_value()

    def set_truncation_value(self):
        """Sets k, which represents how much of the distribution is truncated"""
        weibull = myWeibull()
        k = weibull.cdf(self.lo) + (1.0 - weibull.cdf(self.hi))
        return k

    def get_truncation_value(self):
        return self.k

    def cdf(self, x):
        a = self.a
        b = self.b
        F = 1-np.exp(-(x/b)**a)
        F = 1 / (1.0-self.k) * F  # Account for the truncation
        return F

    def pdf(self, x):
        a = self.a
        b = self.b
        f = a/b * (x/b)**(a-1) * np.exp(-(x/b)**a)
        f = 1 / (1.0-self.k) * f  # Account for the truncation
        return f

    def str(self):
        return "Truncated [%s, %s] weibull(%s, %s)" % (self.lo, self.hi, self.a, self.b)

    def bnd(self):
        return (self.lo, self.hi)


class myWeibull01(object):
    def __init__(self):
        self.a = 1.8
        self.b = 12.552983
        self.lo = 0.0
        self.hi = 30.0/30.

    def cdf(self, x):
        x = x*30
        a = self.a
        b = self.b
        F = 1-np.exp(-(x/b)**a)
        return F

    def pdf(self, x):
        x = x*30
        a = self.a
        b = self.b
        f = a/b * (x/b)**(a-1) * np.exp(-(x/b)**a)
        f = f * 30
        return f

    def mom(self, k):
        # I don't think providing the moments here changes anything down the road when
        # doing PC expansions. Although if I don't pass mom the moments calculated are
        # bad. It appears that they just take the moments assuming a uniform with
        # a range, as that obtained from the bounds.
        # Actually it does affect the PC expansions, this comes into play when getting
        # the orthogonal polynomials.
        a = self.a
        b = self.b
        return b * special.gamma(1.+k*1./a)

    def str(self):
        return "weibull(%s, %s)" % (self.a, self.b)

    def bnd(self):
        return (self.lo, self.hi)


def getWeibull():

    # my_weibull = myWeibull()
    my_weibull = TruncatedWeibull()
    # Set the necessary functions to construct a chaospy distribution
    Weibull = cp.construct(
        cdf=lambda self, x: my_weibull.cdf(x),
        bnd=lambda self: my_weibull.bnd(),
        pdf=lambda self, x: my_weibull.pdf(x),
        # mom=lambda self, k: my_weibull.mom(k),
        str=lambda self: my_weibull.str()
    )

    weibull_dist = Weibull()
    # Dynamically add method
    weibull_dist.get_truncation_value = my_weibull.get_truncation_value
    return weibull_dist


def getWeibull01():

    my_weibull = myWeibull01()
    # Set the necessary functions to construct a chaospy distribution
    Weibull = cp.construct(
        cdf=lambda self, x: my_weibull.cdf(x),
        bnd=lambda self: my_weibull.bnd(),
        pdf=lambda self, x: my_weibull.pdf(x),
        # mom=lambda self, k: my_weibull.mom(k),
        str=lambda self: my_weibull.str()
    )

    weibull_dist = Weibull()
    return weibull_dist


def getWindRose():

    amalia_wind_rose = amaliaWindRose()
    # amalia_wind_rose = amaliaWindRoseRaw()  # Using this option needs updating
    # amalia_wind_rose = amaliaWindRoseRaw01()


    # Set the necessary functions to construct a chaospy distribution
    windRose = cp.construct(
        cdf=lambda self, x: amalia_wind_rose.cdf(x),
        bnd=lambda self: amalia_wind_rose.bnd(),
        pdf=lambda self, x: amalia_wind_rose.pdf(x),
        str=lambda self: amalia_wind_rose.str()
    )

    windrose_dist = windRose()
    # print windrose_dist
    # print windrose_dist.pdf(180)
    # print windrose_dist.pdf(365)
    # print windrose_dist.range()
    return windrose_dist

# amalia_wind_rose = amaliaWindRose()
# x = np.linspace(-0.5, 0.5, 51)
# dx = x[1]-x[0]
# print x+dx/2
# y = amalia_wind_rose._windrose_polyfit(x+dx/2)
# print np.sum(y)*dx
# print y
# imax = np.argmax(y)
# print x[imax]
# print y[imax]
# z = y[imax:-1]
# xz = x[imax:-1]+x[imax]
# z = np.concatenate((y[imax:], y[:imax]))
# print z
# print len(z)
# print z.shape
# print len(y)
# print y.shape
#
# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.plot(x, z)
# plt.show()

# dist = getWindRose()
# x1 = np.linspace(0,360,11)
# print x1
# y1 = dist.pdf(x1)
# print dist._cdf(x1)
# print dist._cdf(360)
# print np.sum(y1)
# plt.figure()
# plt.plot(x1, y1)
# plt.show()





# x = np.linspace(-0.5, 0.5, 361)
# dx = x[1]-x[0]
# a = 140
# b = 470
# xnew = ((b+a)/2. + (b-a)*x)%360
# index = np.argsort(xnew)
# print index
# print xnew[index]

# y = windrose_polyfit(x)
# # print y
# # print np.sum(y*dx)
# fig, ax = plt.subplots()
# ax.plot(x1, y1, label='original')
# ax.plot(xnew[index], y[index]/360., label='smooth')
# # ax.plot(x, y, label='smooth')
# ax.legend()

# # The matlab file PJ sent me
# # Polynomial fit
# # polynomial coefficients
# alpha = np.array([493597.250387841, -207774.160030495, -413203.013010848, 158080.893880027, 127607.500730722, -44242.1722820275, -17735.2623897828, 5422.11156037294, 1057.31910521884, -253.807324825523, -19.8973363502958, 1.43458543839655, 1.05778787373732])
# # alpha = np.array([493597.250387841, -207774.160030495])
# z = np.arange(0, 335*np.pi/180., 0.01)
# # print z
# x = z/(335*np.pi/180.) - 0.5
# # print x
# y = np.polyval(alpha, x)
# plt.figure()
# plt.plot(x, y)



# plt.show()




# x = np.linspace(0,30)
# x = np.linspace(0,30,1001)
# dx= x[1]-x[0]
# print x
# weibull = getWeibull()
# # weibull = cp.weibull(a=0.1)
# y = weibull.pdf(x)
# # y = weibull.pdf(x+0.5)
# print np.sum(y)*dx
# # print y
# a = ''
# for xi in y:
#     a = a + str(xi) + ' '
# # print a
# plt.figure()
# plt.plot(x, y)
#
# plt.show()
# 0.991836543302
# print wind_rose
# nodes, weights = cp.generate_quadrature(order=23, domain=wind_rose, rule="Gaussian")  # Seems like I can only get same optimal points as Dakota! But maybe I can try other rules.
# print nodes
# #print weights
# nodes, weights = cp.generate_quadrature(order=100, domain=wind_rose, rule="Clenshaw")  # Seems like I can only get same optimal points as Dakota! But maybe I can try other rules.
# print nodes
# #print weights
#
#
# print wind_rose_cdf([[360]])
# print wind_rose_cdf([[340,341]])

# print wind_rose.pdf([340, 350])
# # f = wind_rose_func()
# # print f(340), f(350)
#
# print wind_rose.mom(1, order=1001, composite=30.)  # I'm not convinced why these are 180.
# q = cp.variable()
# # print q.keys()
# print cp.E(q,wind_rose)
# # samples = wind_rose.sample(10**3)#, "L")
# # print np.average(samples)
# # print np.mean(samples)
# print wind_rose.sample(4)
# x = np.linspace(0,360,361)
# y = wind_rose.pdf(x)
# p = cp.orth_ttr(2, wind_rose)
# p2 = cp.outer(p, p)
# print 'ttr', cp.E(p2, wind_rose)
# p = cp.orth_bert(2, wind_rose)
# p2 = cp.outer(p, p)
# print 'bert', cp.E(p2, wind_rose)
# p = cp.orth_chol(20, wind_rose)
# p2 = cp.outer(p, p)
# print 'chol', cp.E(p2, wind_rose)
# p = cp.orth_gs(2, wind_rose)
# p2 = cp.outer(p, p)
# print 'gs', cp.E(p2, wind_rose)
#
#
# # print P
# # print cp.E(P[1]*P[2], wind_rose)  # These are not orthogonal, I think I need to define the wind rose nicer.
#
# dist = cp.Gamma(2)
# p = cp.orth_bert(2, dist)
# p2 = cp.outer(p, p)
# print cp.E(p2, dist)
#
# # q = cp.variable()
# # print q
# # dist = cp.Normal()
# # print dist
# # print cp.E(q, dist)
# # print cp.E(q*q, dist)
# # print cp.E(q*q*q*q, dist)
# # print dist.mom((1,2,4))
# # plt.figure()
# # plt.plot(x, y)
# #
# # plt.show()
