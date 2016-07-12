from openmdao.api import Problem, Group, ExternalCode, IndepVarComp, Component
import numpy as np
import os
import chaospy as cp
from getSamplePoints import getSamplePoints


class DakotaStatistics(ExternalCode):
    """Use Dakota to estimate the statistics."""

    def __init__(self, nDirections=10, method_dict=None):
        super(DakotaStatistics, self).__init__()

        # set finite difference options (fd used for testing only)
        # self.fd_options['force_fd'] = True
        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-5
        self.fd_options['step_type'] = 'relative'

        # define inputs
        self.add_param('power', np.zeros(nDirections), units ='kW',
                       desc='vector containing the power production for each winddirection and windspeed pair')
        self.add_param('method_dict', method_dict,
                       desc='parameters for the UQ method')
        self.add_param('weights', np.zeros(nDirections),
                       desc='vector containing the integration weight associated with each power')

        # define output
        self.add_output('mean', val=0.0, units='kWh', desc='mean annual energy output of wind farm')
        self.add_output('std', val=0.0, units='kWh', desc='std of energy output of wind farm')

        # File in which the external code is implemented
        pythonfile = 'getDakotaStatistics.py'
        self.options['command'] = ['python', pythonfile, method_dict['dakota_filename']]

    def solve_nonlinear(self, params, unknowns, resids):

        # Generate the file with the power vector for Dakota
        power = params['power']
        np.savetxt('powerInput.txt', power, header='power')

        # parent solve_nonlinear function actually runs the external code
        super(DakotaStatistics, self).solve_nonlinear(params,unknowns,resids)

        os.remove('powerInput.txt')

        # number of hours in a year
        hours = 8760.0
        # promote statistics to class attribute
        unknowns['mean'] = np.loadtxt('mean.txt')*hours
        unknowns['std'] = np.loadtxt('std.txt')*hours

        # Modify the values for the weibull (speed) case. I need to think about this modification in 2d
        dist = params['method_dict']['distribution']
        if 'weibull' in dist._str():
            bnd = dist.range()
            b = bnd[1][0]  # b=30
            factor = dist._cdf(b)
            unknowns['mean'] = unknowns['mean'] * factor  # weighted by how much of probability is between 0 and 30
            unknowns['std'] = unknowns['std'] * np.sqrt(factor)  # if you look at PC formula for std you see why it is sqrt.

        print 'In DakotaStatistics'

    def linearize(self, params, unknowns, resids):

        J = linearize_function(params)
        # print('Calculate Derivatives:', self.name)

        return J


class ChaospyStatistics(Component):
    """Use chaospy to estimate the statistics."""

    def __init__(self, nDirections=10, method_dict=None):
        super(ChaospyStatistics, self).__init__()

        # set finite difference options (fd used for testing only)
        # self.fd_options['force_fd'] = True
        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-5
        self.fd_options['step_type'] = 'relative'

        # define inputs
        self.add_param('power', np.zeros(nDirections), units ='kW',
                       desc='vector containing the power production for each winddirection and windspeed pair')
        self.add_param('method_dict', method_dict,
                       desc='parameters for the UQ method')
        self.add_param('weights', np.zeros(nDirections),
                       desc='vector containing the integration weight associated with each power')

        # define output
        self.add_output('mean', val=0.0, units='kWh', desc='mean annual energy output of wind farm')
        self.add_output('std', val=0.0, units='kWh', desc='std of energy output of wind farm')


    def solve_nonlinear(self, params, unknowns, resids):

        power = params['power']
        method_dict = params['method_dict']
        dist = method_dict['distribution']
        n = len(power)
        points, weights = cp.generate_quadrature(order=n-1, domain=dist, rule='G')
        poly = cp.orth_ttr(n-1, dist)  # Think about the n-1
        # Double check if giving me orthogonal polynomials
        # p2 = cp.outer(poly, poly)
        # norms = np.diagonal(cp.E(p2, dist))
        # print 'diag', norms

        # expansion, coeff = cp.fit_quadrature(poly, points, weights, power, retall=True, norms=norms)
        expansion, coeff = cp.fit_quadrature(poly, points, weights, power, retall=True)
        # expansion, coeff = cp.fit_regression(poly, points, power, retall=True)

        mean = cp.E(expansion, dist, rule='G')
        # print 'mean cp.E =', mean
        # # mean = sum(power*weights)
        # print 'mean sum =', sum(power*weights)
        # print 'mean coeff =', coeff[0]*8760/1e6
        std = cp.Std(expansion, dist, rule='G')

        # print mean
        # print std
        # print np.sqrt(np.sum(coeff[1:]**2 * cp.E(poly**2, dist)[1:]))
        # # std = np.sqrt(np.sum(coeff[1:]**2 * cp.E(poly**2, dist)[1:]))
        # number of hours in a year
        hours = 8760.0
        # promote statistics to class attribute
        unknowns['mean'] = mean*hours
        unknowns['std'] = std*hours

        print 'In ChaospyStatistics'

    def linearize(self, params, unknowns, resids):

        # Todo update linearize
        J = linearize_function(params)
        return J


class RectStatistics(Component):
    """Use simple rectangle integration to estimate the statistics."""

    def __init__(self, nDirections=10, method_dict=None):

        super(RectStatistics, self).__init__()

        # set finite difference options (fd used for testing only)
        # self.fd_options['force_fd'] = True
        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-5
        self.fd_options['step_type'] = 'relative'

        # define inputs
        self.add_param('power', np.zeros(nDirections), units ='kW',
                       desc='vector containing the power production for each winddirection and windspeed pair')
        self.add_param('method_dict', method_dict,
                       desc='parameters for the UQ method')
        self.add_param('weights', np.zeros(nDirections),
                       desc='vector containing the integration weight associated with each power')

        # define output
        self.add_output('mean', val=0.0, units='kWh', desc='mean annual energy output of wind farm')
        self.add_output('std', val=0.0, units='kWh', desc='std of energy output of wind farm')

    def solve_nonlinear(self, params, unknowns, resids):

        power = params['power']
        weights = params['weights']

        mean = sum(power*weights)
        # Calculate std to ensure it is positive, first method could have issues for small number of samples
        # std = np.sqrt(sum(np.power(power, 2)*weights) - np.power(mean, 2))  # Revisar if this is right
        var = np.sum(np.power(power - mean, 2) * weights)
        std = np.sqrt(var)

        # number of hours in a year
        hours = 8760.0
        # promote statistics to class attribute
        unknowns['mean'] = mean*hours
        unknowns['std'] = std*hours
        # print weights
        # print unknowns['mean']

        print 'In RectStatistics'

    def linearize(self, params, unknowns, resids):

        J = linearize_function(params)
        # print('Calculate Derivatives:', self.name)
        return J


def linearize_function(params):

    weights = params['weights']

    # number of hours in a year
    hours = 8760.0
    dmean_dpower = weights*hours

    J = {}
    J[('mean', 'power')] = np.array([dmean_dpower])

    return J


if __name__ == "__main__":

    import distributions
    from statistics_convergence import getPoints
    method_dict = {}
    method_dict['method']           = 'rect'
    method_dict['uncertain_var']    = 'direction'
    method_dict['layout']           = 'optimized'

    if method_dict['uncertain_var'] == 'speed':
        dist = distributions.getWeibull()
        method_dict['distribution'] = dist
    elif method_dict['uncertain_var'] == 'direction':
        dist = distributions.getWindRose()
        method_dict['distribution'] = dist
    else:
        raise ValueError('unknown uncertain_var option "%s", valid options "speed" or "direction".' %method_dict['uncertain_var'])

    method_dict['dakota_filename'] = 'dakotageneral.in'

    n = 10
    unused, weights = getPoints(method_dict, n)
    prob = Problem(root=Group())
    prob.root.add('p', IndepVarComp('power', np.random.rand(n)))
    prob.root.add('w', IndepVarComp('weight', weights))
    if method_dict['method'] == 'rect':
        prob.root.add('AEPComp', RectStatistics(nDirections=n, method_dict=method_dict))#, promotes=['*'])  # No need to promote because of the explicit connection below
    if method_dict['method'] == 'dakota':
        prob.root.add('AEPComp', DakotaStatistics(nDirections=n, method_dict=method_dict))#, promotes=['*'])
    prob.root.connect('p.power', 'AEPComp.power')
    prob.root.connect('w.weight', 'AEPComp.weights')
    prob.setup()
    prob.run()
    print 'AEP = ', (prob.root.AEPComp.unknowns['mean'])
    print 'power = ', (prob.root.AEPComp.params['power'])
    print prob.root.AEPComp.params.keys()
    # J = prob.calc_gradient(['AEPComp.mean'], ['p.power'])  # I'm not sure why this returns zero
    # print 'power directions gradient = ', J

    # This check works
    data = prob.check_partial_derivatives()
