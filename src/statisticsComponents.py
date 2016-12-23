from openmdao.api import Problem, Group, ExternalCode, IndepVarComp, Component
import numpy as np
import os
import json
import shutil
import chaospy as cp
from getSamplePoints import getSamplePoints


class DakotaStatistics(ExternalCode):
    """Use Dakota to estimate the statistics."""

    def __init__(self, nTurbines=60, nDirections=10, method_dict=None):
        super(DakotaStatistics, self).__init__()

        # set finite difference options (fd used for testing only)
        # self.deriv_options['type'] = 'fd'
        # self.deriv_options['form'] = 'central'
        # self.deriv_options['step_size'] = 1.0e-5
        # self.deriv_options['step_calc'] = 'relative'

        # define inputs
        self.add_param('Powers', np.zeros(nDirections), units ='kW',
                       desc='vector containing the power production for each winddirection and windspeed pair')
        self.add_param('method_dict', method_dict,
                       desc='parameters for the UQ method')
        self.add_param('windWeights', np.zeros(nDirections),
                       desc='vector containing the integration weight associated with each power')
        self.add_param('turbineX', np.zeros(nTurbines),
                       desc='vector containing the turbine X locations')
        self.add_param('turbineY', np.zeros(nTurbines),
                       desc='vector containing the turbine Y locations')
        self.add_param('dpower_dturbX', np.zeros([nDirections, nTurbines]),
                       desc='vector containing the gradient of the power wrt turbineX locations')
        self.add_param('dpower_dturbY', np.zeros([nDirections, nTurbines]),
                       desc='vector containing the gradient of the power wrt turbineY locations')

        # define output
        self.add_output('mean', val=0.0, units='kWh', desc='mean annual energy output of wind farm')
        self.add_output('std', val=0.0, units='kWh', desc='std of energy output of wind farm')

        # File in which the external code is implemented
        pythonfile = 'getDakotaStatistics.py'
        self.options['command'] = ['python', pythonfile, method_dict['dakota_filename']]

    def solve_nonlinear(self, params, unknowns, resids):

        # Generate the file with the power vector for Dakota
        power = params['Powers']
        np.savetxt('powerInput.txt', power, header='Powers')

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

        dpower_dturbX = params['dpower_dturbX']
        dpower_dturbY = params['dpower_dturbY']
        weights = params['windWeights']

        m, n = dpower_dturbX.shape
        dmean_dturbX = np.zeros([1, n])
        dmean_dturbY = np.zeros([1, n])

        for j in range(n):
            for i in range(m):
                dmean_dturbX[0][j] += weights[i]*dpower_dturbX[i][j]
                dmean_dturbY[0][j] += weights[i]*dpower_dturbY[i][j]

        # number of hours in a year
        hours = 8760.0

        J = {}
        J[('mean', 'turbineX')] = hours*dmean_dturbX
        J[('mean', 'turbineY')] = hours*dmean_dturbY

        # J = linearize_function(params)
        # print('Calculate Derivatives:', self.name)

        return J


class ChaospyStatistics(Component):
    """Use chaospy to estimate the statistics."""

    def __init__(self, nDirections=10, method_dict=None):
        super(ChaospyStatistics, self).__init__()

        # set finite difference options (fd used for testing only)
        # self.deriv_options['force_fd'] = True
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1.0e-5
        self.deriv_options['step_calc'] = 'relative'

        # define inputs
        self.add_param('Powers', np.zeros(nDirections), units ='kW',
                       desc='vector containing the power production for each winddirection and windspeed pair')
        self.add_param('method_dict', method_dict,
                       desc='parameters for the UQ method')
        self.add_param('windWeights', np.zeros(nDirections),
                       desc='vector containing the integration weight associated with each power')

        # define output
        self.add_output('mean', val=0.0, units='kWh', desc='mean annual energy output of wind farm')
        self.add_output('std', val=0.0, units='kWh', desc='std of energy output of wind farm')


    def solve_nonlinear(self, params, unknowns, resids):

        power = params['Powers']
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

    def __init__(self, nTurbines=60, nDirections=10, method_dict=None):

        super(RectStatistics, self).__init__()

        # set finite difference options (fd used for testing only)
        # self.deriv_options['force_fd'] = True
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1.0e-5
        self.deriv_options['step_calc'] = 'relative'

        # define inputs
        self.add_param('Powers', np.zeros(nDirections), units ='kW',
                       desc='vector containing the power production for each winddirection and windspeed pair')
        self.add_param('method_dict', method_dict,
                       desc='parameters for the UQ method')
        self.add_param('windWeights', np.zeros(nDirections),
                       desc='vector containing the integration weight associated with each power')
        self.add_param('turbineX', np.zeros(nTurbines),
                       desc='vector containing the turbine X locations')
        self.add_param('turbineY', np.zeros(nTurbines),
                       desc='vector containing the turbine Y locations')


        # define output
        self.add_output('mean', val=0.0, units='kWh', desc='mean annual energy output of wind farm')
        self.add_output('std', val=0.0, units='kWh', desc='std of energy output of wind farm')

    def solve_nonlinear(self, params, unknowns, resids):

        power = params['Powers']
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
        # This was added to make the optimization video.
        # print 'Print turbine locations'
        # print '\tturbineX \t turbineY'
        # for tX, tY in zip(params['turbineX'], params['turbineY']):
        #     print '%.2f' % tX, '\t', '%.2f' % tY
        # try:
        #     f = open('turbinelocations.json', 'r')  # Make sure this file is not here when I start running.
        #     r = json.load(f)
        #     f.close()
        #
        #     key = str(int(max([int(i) for i in r.keys()])) + 1)
        #     r[key] = {'turbineX': params['turbineX'].tolist(), 'turbineY': params['turbineY'].tolist()}
        #     f = open('turbinelocationstemp.json', 'w')
        #     json.dump(r, f, indent=2)
        #     f.close()
        #     shutil.move('turbinelocationstemp.json', 'turbinelocations.json')
        #
        #
        # except IOError:
        #     print 'I caught the exception'
        #     obj = {'0': {'turbineX': params['turbineX'].tolist(), 'turbineY': params['turbineY'].tolist()}}
        #
        #     jsonfile = open('turbinelocations.json', 'w')
        #     print json.dumps(obj, indent=2)
        #     json.dump(obj, jsonfile, indent=2)
        #     jsonfile.close()


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
    J[('mean', 'Powers')] = np.array([dmean_dpower])

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
    prob.root.add('p', IndepVarComp('Powers', np.random.rand(n)))
    prob.root.add('w', IndepVarComp('windWeights', weights))
    if method_dict['method'] == 'rect':
        prob.root.add('AEPComp', RectStatistics(nDirections=n, method_dict=method_dict))#, promotes=['*'])  # No need to promote because of the explicit connection below
    if method_dict['method'] == 'dakota':
        prob.root.add('AEPComp', DakotaStatistics(nDirections=n, method_dict=method_dict))#, promotes=['*'])
    prob.root.connect('p.Powers', 'AEPComp.Powers')
    prob.root.connect('w.windWeights', 'AEPComp.windWeights')
    prob.setup()
    prob.run()
    print 'AEP = ', (prob.root.AEPComp.unknowns['mean'])
    print 'power = ', (prob.root.AEPComp.params['Powers'])
    print prob.root.AEPComp.params.keys()
    # J = prob.calc_gradient(['AEPComp.mean'], ['p.power'])  # I'm not sure why this returns zero
    # print 'power directions gradient = ', J

    # This check works
    data = prob.check_partial_derivatives()
