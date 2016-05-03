import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import special


class amaliaWindRose(object):
    # Fill out this class
    def __init__(self):
        self.lo = 0.0
        self.hi = 360.0
        self.inputfile = 'amalia_windrose_8.txt'

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
        f = self._wind_rose_func()
        return f(x)

    def cdf(self, x):
        # Integrate by rectangle rule
        dx = 0.001  # interval spacing
        f = self._wind_rose_func()
        cdf = []
        for x_i in x[0]:
            X = np.arange(dx/2,x_i,dx)
            cdf.append(np.sum(f(X)*dx))  # integration by rectangle rule.
            # cdf.append(np.sum(X*f(X)*dx))  # I can use this to calculate the mean, when x is 360
        return np.array(cdf)

    def str(self):
        return "Amalia windrose"

    def bnd(self):
        return (self.lo, self.hi)


class myWeibull(object):
    def __init__(self):
        self.a = 1.8
        self.b = 12.552983
        # self.a = 0.1 #1.8
        # self.b = 1.0 #12.552983
        self.lo = 0.0
        self.hi = 30.0

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


def getWeibull():

    my_weibull = myWeibull()
    # Set the necessary functions to construct a chaospy distribution
    Weibull = cp.construct(
        cdf = lambda self, x: my_weibull.cdf(x),
        bnd = lambda self: my_weibull.bnd(),
        pdf = lambda self, x: my_weibull.pdf(x),
        # mom = lambda self, k: my_weibull.mom(k),
        str = lambda self: my_weibull.str()
    )

    weibull_dist = Weibull()
    # print weibull_dist
    # print weibull_dist.pdf(31)
    # print weibull_dist.range()
    # print weibull_dist.mom((1,2))
    return weibull_dist

def getWindRose():

    amalia_wind_rose = amaliaWindRose()
    # Set the necessary functions to construct a chaospy distribution
    windRose = cp.construct(
        cdf = lambda self, x: amalia_wind_rose.cdf(x),
        bnd = lambda self: amalia_wind_rose.bnd(),
        pdf = lambda self, x: amalia_wind_rose.pdf(x),
        str = lambda self: amalia_wind_rose.str()
    )

    windrose_dist = windRose()
    # print windrose_dist
    # print windrose_dist.pdf(180)
    # print windrose_dist.pdf(365)
    # print windrose_dist.range()
    return windrose_dist


# def windrose_polyfit(x):
#     y = 493597.250387841 + \
#         -207774.160030495*x #+ \
#         # -413203.013010848*np.power(x, 2) + \
#         # 158080.893880027*np.power(x, 3) + \
#         # 127607.500730722*np.power(x, 4) + \
#         # -44242.1722820275*np.power(x, 5) + \
#         # -17735.2623897828*np.power(x, 6) + \
#         # 5422.11156037294*np.power(x, 7) + \
#         # 1057.31910521884*np.power(x, 8) + \
#         # -253.807324825523*np.power(x, 9) + \
#         # -19.8973363502958*np.power(x, 10) + \
#         # 1.43458543839655*np.power(x, 11) + \
#         # 1.05778787373732*np.power(x, 12)
#     return y

def windrose_polyfit(x):
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

# dist = getWindRose()
# x = np.linspace(0,360,361)
# y = dist.pdf(x)
# # print np.sum(y)
# plt.figure()
# plt.plot(x, y)
#
x = np.linspace(-0.5, 0.5, 101)
print x
a = 140
b = 470
xnew = ((b+a)/2. + (b-a)*x)%360
print xnew
print np.sort(xnew)
index = np.argsort(xnew)
print xnew[index]

y = windrose_polyfit(x)
print y
plt.figure()
plt.plot(xnew[index], y[index],  'o-')


# The matlab file PJ sent me
# Polynomial fit
# polynomial coefficients
alpha = np.array([493597.250387841, -207774.160030495, -413203.013010848, 158080.893880027, 127607.500730722, -44242.1722820275, -17735.2623897828, 5422.11156037294, 1057.31910521884, -253.807324825523, -19.8973363502958, 1.43458543839655, 1.05778787373732])
# alpha = np.array([493597.250387841, -207774.160030495])
z = np.arange(0, 335*np.pi/180., 0.01)
# print z
x = z/(335*np.pi/180.) - 0.5
# print x
y = np.polyval(alpha, x)
plt.figure()
plt.plot(x, y)


# a = np.array([5,4,3,2,1])
# b = np.array([5,4,3,2,1])
# c = np.array([a,b])
# print c
# print c[0,:]
# # print c[c[0,:].argsort()]




plt.show()




# x = np.linspace(0,30)
# x = np.linspace(0,30,31)
# print x
# weibull = getWeibull()
# # weibull = cp.weibull(a=0.1)
# y = weibull.pdf(x)
# print np.sum(y)
# print y
# a = ''
# for xi in y:
#     a = a + str(xi) + ' '
# print a
# plt.figure()
# plt.plot(x, y)
#
# plt.show()

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