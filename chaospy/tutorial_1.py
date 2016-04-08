import chaospy as cp
import numpy as np


def u(x, a):
    return np.exp(-a*x)


dist_a = cp.Uniform(0, 0.1)

samples_a = dist_a.sample(size=1000)

x = np.linspace(0, 10, 100)

samples_u = [u(x, a) for a in samples_a]

E = np.mean(samples_u, 0)
Var = np.var(samples_u, 0)

print E
print Var