from openmdao.api import Problem, Group, ExternalCode, IndepVarComp, Component
import numpy as np
import os
import chaospy as cp
import distributions
from getSamplePoints import getSamplePoints


class DakotaStatistics(ExternalCode):
    """Use Dakota to estimate the AEP based on weighted power production."""

    def __init__(self, nDirections=10, dakotaFileName='dakotaAEP.in'):
        super(DakotaStatistics, self).__init__()

        # set finite difference options (fd used for testing only)
        # self.fd_options['force_fd'] = True
        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-5
        self.fd_options['step_type'] = 'relative'

        # define inputs
        self.add_param('power', np.zeros(nDirections), units ='kW',
                       desc = 'vector containing the power production at each wind direction ccw from north')

        # define output
        self.add_output('mean', val=0.0, units='kWh', desc='mean annual energy output of wind farm')
        self.add_output('std', val=0.0, units='kWh', desc='std of energy output of wind farm')

        # File in which the external code is implemented
        pythonfile = 'getDakotaStatistics.py'
        self.options['command'] = ['python', pythonfile, dakotaFileName]

    def solve_nonlinear(self, params, unknowns, resids):

        # Generate the file with the power vector for Dakota
        power = params['power']
        np.savetxt('powerInput.txt', power, header='power')

        # parent solve_nonlinear function actually runs the external code
        super(DakotaStatistics, self).solve_nonlinear(params,unknowns,resids)

        os.remove('powerInput.txt')

        # Read in the calculated AEP
        # number of hours in a year
        hours = 8760.0
        unknowns['mean'] = np.loadtxt('mean.txt')*hours
        unknowns['std'] = np.loadtxt('std.txt')*hours

        print 'In DakotaStatistics'

    def linearize(self, params, unknowns, resids):

        # Todo update linearize
        J = linearize_function(params)
        return J


class ChaospyStatistics(Component):
    """Use chaospy to estimate the statistics."""

    def __init__(self, nDirections=10):
        super(ChaospyStatistics, self).__init__()

        # set finite difference options (fd used for testing only)
        # self.fd_options['force_fd'] = True
        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-5
        self.fd_options['step_type'] = 'relative'

        # define inputs
        self.add_param('power', np.zeros(nDirections), units ='kW',
                       desc = 'vector containing the power production at each wind direction ccw from north')

        # define output
        self.add_output('mean', val=0.0, units='kWh', desc='mean annual energy output of wind farm')
        self.add_output('std', val=0.0, units='kWh', desc='std of energy output of wind farm')


    def solve_nonlinear(self, params, unknowns, resids):

        power = params['power']
        n = len(power)
        # windrose_dist = distributions.getWindRose()
        weibull_dist = distributions.getWeibull()
        # weibull_dist = cp.weibull(a=0.1)
        print 'aloha', weibull_dist
        # weibull_dist = distributions.getWindRose()
        points, weights = cp.generate_quadrature(order=n-1, domain=weibull_dist, rule="Clenshaw")
        # points, weights = cp.generate_quadrature(order=n, domain=weibull_dist, rule="L")
        # points, weights = trapezoid(n-1)
        # points, weights = distributions.rectangle(n)
        poly = cp.orth_chol(n-1, weibull_dist)  # double check this is giving me good orthogonal polynomials.
        # poly = cp.orth_gs(n-1, weibull_dist)
        print poly, '\n'
        p2 = cp.outer(poly, poly)
        print 'chol', cp.E(p2, weibull_dist)
        norms = np.diagonal(cp.E(p2, weibull_dist))
        print 'diag', norms

        expansion, coeff = cp.fit_quadrature(poly, points, weights, power, retall=True, norms=norms)
        # expansion, coeff = cp.fit_quadrature(poly, points, weights, power, retall=True)

        mean = cp.E(expansion, weibull_dist)
        # print poly[0]
        # print cp.E(poly[0]*poly[0], weibull_dist)
        print 'mean cp.E =', mean
        # mean = sum(power*weights)
        print 'mean sum =', mean
        print 'mean coeff =', coeff[0]
        std = cp.Std(expansion, weibull_dist)

        print mean
        print std
        print np.sqrt(np.sum(coeff[1:]**2 * cp.E(poly**2, weibull_dist)[1:]))

        # number of hours in a year
        hours = 8760.0
        unknowns['mean'] = mean*hours
        unknowns['std'] = std*hours

        print 'In ChaospyStatistics'

    def linearize(self, params, unknowns, resids):

        # Todo update linearize
        J = linearize_function(params)
        return J


class RectStatistics(Component):
    """Use simple integration to estimate the AEP based on weighted power production."""

    def __init__(self, nDirections=10):

        super(RectStatistics, self).__init__()

        # set finite difference options (fd used for testing only)
        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-5
        self.fd_options['step_type'] = 'relative'

        # define inputs
        self.add_param('power', np.zeros(nDirections), units ='kW',
                       desc = 'vector containing the power production at each wind direction ccw from north')
        self.add_param('weights', np.zeros(nDirections),
                       desc = 'vector containing the weights for integration.')
        self.add_param('frequency', np.zeros(nDirections),
                       desc = 'vector containing the frequency from the probability density function.')

        # define output
        self.add_output('AEP', val=0.0, units='kWh', desc='total annual energy output of wind farm')
        self.add_output('std_energy', val=0.0, units='kWh', desc='standard deviation of energy output of wind farm')


    def solve_nonlinear(self, params, unknowns, resids):

        power = params['power']
        rho = params['frequency']
        weight = params['weights'] # The weights of the integration points

        # number of hours in a year
        hours = 8760.0

        # calculate the statistics
        mean = sum(power*weight*rho)
        print 'first term = ', sum(np.power(power, 2)*weight*rho)/1e9
        print 'second term = ', np.power(mean, 2)/1e9
        std = np.sqrt(sum(np.power(power, 2)*weight*rho) - np.power(mean, 2))
        AEP = mean*hours
        std_energy = std*hours

        # promote AEP result to class attribute
        unknowns['AEP'] = AEP
        unknowns['std_energy'] = std_energy

        print 'In RectStatistics'

    def linearize(self, params, unknowns, resids):

        J = linearize_function(params)
        return J


def linearize_function(params):

    power = params['power']
    weight = params['weights'] # The weights of the integration points
    rho = params['frequency']
    # number of hours in a year
    hours = 8760.0
    dAEP_dpower = weight*rho*hours
    dAEP_dweight = power*rho*hours
    dAEP_drho = power*weight*hours

    J = {}
    J[('AEP', 'power')] = np.array([dAEP_dpower])
    J[('AEP', 'weights')] = np.array([dAEP_dweight])
    J[('AEP', 'frequency')] = np.array([dAEP_drho])

    return J


if __name__ == "__main__":

    from WindFreqFunctions import wind_direction_pdf
    dakotaFileName = 'dakotaAEPdirection.in'
    winddirections, weights = getSamplePoints(dakotaFileName)
    f = wind_direction_pdf()
    rho = f(winddirections)
    prob = Problem(root=Group())
    prob.root.add('p', IndepVarComp('power', np.random.rand(10)))
    prob.root.add('w', IndepVarComp('weight', weights))
    prob.root.add('rho', IndepVarComp('frequency', rho))
    # prob.root.add('DakotaAEP', DakotaAEP(dakotaFileName=dakotaFileName))
    prob.root.add('DakotaAEP', SimpleAEP())
    prob.root.connect('p.power', 'DakotaAEP.power')
    prob.root.connect('w.weight', 'DakotaAEP.weights')
    prob.root.connect('rho.frequency', 'DakotaAEP.frequency')
    prob.setup()
    prob.run()
    print 'AEP = ', (prob.root.DakotaAEP.unknowns['AEP'])
    print 'power directions = ', (prob.root.DakotaAEP.params['power'])
    print prob.root.DakotaAEP.params.keys()
    # The DakotaAEP.power_directions key is not recognized
    # J = prob.calc_gradient(['DakotaAEP.AEP'], ['DakotaAEP.power'])
    # J = prob.calc_gradient(['DakotaAEP.AEP'], ['p.power'])
    # print 'power directions gradient = ', J

    # This check works
    data = prob.check_partial_derivatives()
