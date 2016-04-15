from openmdao.api import Problem, Group, ExternalCode, IndepVarComp, Component
import numpy as np
import os
import chaospy as cp
from getSamplePoints import getSamplePoints
import quadrature_rules


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
                       desc = 'vector containing the power production at each wind direction ccw from north')

        # define output
        self.add_output('mean', val=0.0, units='kWh', desc='mean annual energy output of wind farm')
        self.add_output('std', val=0.0, units='kWh', desc='std of energy output of wind farm')

        # File in which the external code is implemented
        pythonfile = 'getDakotaStatistics.py'
        self.options['command'] = ['python', pythonfile, method_dict['filename']]

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

        print 'In DakotaStatistics'

    def linearize(self, params, unknowns, resids):

        # Todo update linearize
        J = linearize_function(params)
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
                       desc = 'vector containing the power production at each wind direction ccw from north')
        self.add_param('method_dict', method_dict,
                       desc = 'parameters for the UQ method')

        # define output
        self.add_output('mean', val=0.0, units='kWh', desc='mean annual energy output of wind farm')
        self.add_output('std', val=0.0, units='kWh', desc='std of energy output of wind farm')


    def solve_nonlinear(self, params, unknowns, resids):

        power = params['dirPowers']
        method_dict = params['method_dict']
        dist = method_dict['distribution']
        rule = method_dict['rule']
        n = len(power)
        if rule != 'rectangle':
            points, weights = cp.generate_quadrature(order=n-1, domain=dist, rule=rule)
        else:
            points, weights = quadrature_rules.rectangle(n, method_dict['distribution'])

        poly = cp.orth_chol(n-1, dist)
        # poly = cp.orth_bert(n-1, dist)
        # double check this is giving me good orthogonal polynomials.
        # print poly, '\n'
        p2 = cp.outer(poly, poly)
        # print 'chol', cp.E(p2, dist)
        norms = np.diagonal(cp.E(p2, dist))
        print 'diag', norms

        expansion, coeff = cp.fit_quadrature(poly, points, weights, power, retall=True, norms=norms)
        # expansion, coeff = cp.fit_quadrature(poly, points, weights, power, retall=True)

        mean = cp.E(expansion, dist)
        print 'mean cp.E =', mean
        # mean = sum(power*weights)
        print 'mean sum =', sum(power*weights)
        print 'mean coeff =', coeff[0]
        std = cp.Std(expansion, dist)

        print mean
        print std
        print np.sqrt(np.sum(coeff[1:]**2 * cp.E(poly**2, dist)[1:]))
        # std = np.sqrt(np.sum(coeff[1:]**2 * cp.E(poly**2, dist)[1:]))
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
        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-5
        self.fd_options['step_type'] = 'relative'

        # define inputs
        self.add_param('dirPowers', np.zeros(nDirections), units ='kW',
                       desc = 'vector containing the power production at each wind direction ccw from north')
        self.add_param('method_dict', method_dict,
                       desc = 'parameters for the UQ method')
        self.add_param('weights', np.zeros(nDirections), desc = 'weight assigned to each wind direction when integrating')
        self.add_param('frequency', np.zeros(nDirections), desc = 'the frequency to wind from each direction')

        # define output
        self.add_output('mean', val=0.0, units='kWh', desc='mean annual energy output of wind farm')
        self.add_output('std', val=0.0, units='kWh', desc='std of energy output of wind farm')

    def solve_nonlinear(self, params, unknowns, resids):

        power = params['dirPowers']
        n = len(power)
        method_dict = params['method_dict']
        dist = method_dict['distribution']
        unused, weights = quadrature_rules.rectangle(n, method_dict['distribution'])

        mean = sum(power*weights)
        std = np.sqrt(sum(np.power(power, 2)*weights) - np.power(mean, 2))  # Revisar if this is right

        # number of hours in a year
        hours = 8760.0
        # promote statistics to class attribute
        unknowns['mean'] = mean*hours
        unknowns['std'] = std*hours

        print 'In RectStatistics'

    def linearize(self, params, unknowns, resids):

        J = linearize_function(params)
        return J


def linearize_function(params):

    power = params['dirPowers']
    weight = params['weights'] # The weights of the integration points
    rho = params['frequency']
    # number of hours in a year
    hours = 8760.0
    dAEP_dpower = weight*rho*hours
    dAEP_dweight = power*rho*hours
    dAEP_drho = power*weight*hours

    J = {}
    J[('AEP', 'dirPowers')] = np.array([dAEP_dpower])
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
    prob.root.add('p', IndepVarComp('dirPowers', np.random.rand(10)))
    prob.root.add('w', IndepVarComp('weight', weights))
    prob.root.add('rho', IndepVarComp('frequency', rho))
    # prob.root.add('DakotaAEP', DakotaAEP(dakotaFileName=dakotaFileName))
    prob.root.add('DakotaAEP', SimpleAEP())
    prob.root.connect('p.dirPowers', 'DakotaAEP.dirPowers')
    prob.root.connect('w.weight', 'DakotaAEP.weights')
    prob.root.connect('rho.frequency', 'DakotaAEP.frequency')
    prob.setup()
    prob.run()
    print 'AEP = ', (prob.root.DakotaAEP.unknowns['AEP'])
    print 'power directions = ', (prob.root.DakotaAEP.params['dirPowers'])
    print prob.root.DakotaAEP.params.keys()
    # The DakotaAEP.power_directions key is not recognized
    # J = prob.calc_gradient(['DakotaAEP.AEP'], ['DakotaAEP.power'])
    # J = prob.calc_gradient(['DakotaAEP.AEP'], ['p.power'])
    # print 'power directions gradient = ', J

    # This check works
    data = prob.check_partial_derivatives()
