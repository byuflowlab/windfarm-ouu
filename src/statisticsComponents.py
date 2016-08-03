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
        self.add_param('dirPowers', np.zeros(nDirections), units ='kW',
                       desc='vector containing the power production for each winddirection and windspeed pair')
        self.add_param('method_dict', method_dict,
                       desc='parameters for the UQ method')
        self.add_param('windWeights', np.zeros(nDirections),
                       desc='vector containing the integration weight associated with each power')

        # define output
        self.add_output('mean', val=0.0, units='kWh', desc='mean annual energy output of wind farm')
        self.add_output('std', val=0.0, units='kWh', desc='std of energy output of wind farm')

        # File in which the external code is implemented
        pythonfile = 'getDakotaStatistics.py'
        self.options['command'] = ['python', pythonfile, method_dict['dakota_filename']]

    def solve_nonlinear(self, params, unknowns, resids):

        # Generate the file with the power vector for Dakota
        power = params['dirPowers']
        np.savetxt('powerInput.txt', power, header='dirPowers')

        # parent solve_nonlinear function actually runs the external code
        super(DakotaStatistics, self).solve_nonlinear(params,unknowns,resids)

        os.remove('powerInput.txt')

        # number of hours in a year
        hours = 8760.0
        # promote statistics to class attribute
        unknowns['mean'] = np.loadtxt('mean.txt')*hours
        unknowns['std'] = np.loadtxt('std.txt')*hours

        # Modify the statistics to account for the truncation of the weibull (speed) case.
        modify_statistics(params, unknowns)  # It doesn't do anything for the direction case.

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
        self.add_param('dirPowers', np.zeros(nDirections), units ='kW',
                       desc='vector containing the power production for each winddirection and windspeed pair')
        self.add_param('method_dict', method_dict,
                       desc='parameters for the UQ method')
        self.add_param('windWeights', np.zeros(nDirections),
                       desc='vector containing the integration weight associated with each power')

        # define output
        self.add_output('mean', val=0.0, units='kWh', desc='mean annual energy output of wind farm')
        self.add_output('std', val=0.0, units='kWh', desc='std of energy output of wind farm')


    def solve_nonlinear(self, params, unknowns, resids):

        power = params['dirPowers']
        method_dict = params['method_dict']
        dist = method_dict['distribution']
        n = len(power)
        points, weights = cp.generate_quadrature(order=n-1, domain=dist, rule='G')
        poly = cp.orth_ttr(n-1, dist)  # Think about the n-1 for 1d for 2d or more it would be n-2. Details Dakota reference manual quadrature order.
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

        # Modify the statistics to account for the truncation of the weibull (speed) case.
        modify_statistics(params, unknowns)  # It doesn't do anything for the direction case.

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
        self.add_param('dirPowers', np.zeros(nDirections), units ='kW',
                       desc='vector containing the power production for each winddirection and windspeed pair')
        self.add_param('method_dict', method_dict,
                       desc='parameters for the UQ method')
        self.add_param('windWeights', np.zeros(nDirections),
                       desc='vector containing the integration weight associated with each power')

        # define output
        self.add_output('mean', val=0.0, units='kWh', desc='mean annual energy output of wind farm')
        self.add_output('std', val=0.0, units='kWh', desc='std of energy output of wind farm')

    def solve_nonlinear(self, params, unknowns, resids):

        power = params['dirPowers']
        weights = params['windWeights']

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

        # Modify the statistics to account for the truncation of the weibull (speed) case.
        modify_statistics(params, unknowns)  # It doesn't do anything for the direction case.

        print 'In RectStatistics'

    def linearize(self, params, unknowns, resids):

        J = linearize_function(params)
        # print('Calculate Derivatives:', self.name)
        return J


def linearize_function(params):

    weights = params['windWeights']

    # number of hours in a year
    hours = 8760.0
    dmean_dpower = weights*hours

    J = {}
    J[('mean', 'dirPowers')] = np.array([dmean_dpower])

    return J


def modify_statistics(params, unknowns):
    uncertain_var = params['method_dict']['uncertain_var']
    if uncertain_var == 'direction':
        pass
    else:  # either the speed or the speed and direction case
        dist = params['method_dict']['distribution']
        if uncertain_var == 'speed':
            k = dist.get_truncation_value()  # how much of the probability was truncated
        if uncertain_var == 'direction_and_speed':
            dist = dist[1]
            k = dist.get_truncation_value()  # how much of the probability was truncated
        meant = unknowns['mean']  # the truncated mean
        stdt = unknowns['std']  # the truncated std
        unknowns['mean'] = (1-k) * meant  # weighted by how much of probability is between 0 and 30 or a and b
        unknowns['std'] = np.sqrt(1-k) * stdt + np.sqrt(k*(1-k)) * meant  # formula found in truncation write up.

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
    prob.root.add('p', IndepVarComp('dirPowers', np.random.rand(n)))
    prob.root.add('w', IndepVarComp('windWeights', weights))
    if method_dict['method'] == 'rect':
        prob.root.add('AEPComp', RectStatistics(nDirections=n, method_dict=method_dict))#, promotes=['*'])  # No need to promote because of the explicit connection below
    if method_dict['method'] == 'dakota':
        prob.root.add('AEPComp', DakotaStatistics(nDirections=n, method_dict=method_dict))#, promotes=['*'])
    prob.root.connect('p.dirPowers', 'AEPComp.dirPowers')
    prob.root.connect('w.windWeights', 'AEPComp.windWeights')
    prob.setup()
    prob.run()
    print 'AEP = ', (prob.root.AEPComp.unknowns['mean'])
    print 'power = ', (prob.root.AEPComp.params['dirPowers'])
    print prob.root.AEPComp.params.keys()
    # J = prob.calc_gradient(['AEPComp.mean'], ['p.power'])  # I'm not sure why this returns zero
    # print 'power directions gradient = ', J

    # This check works
    data = prob.check_partial_derivatives()
