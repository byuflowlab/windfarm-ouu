from openmdao.api import Problem, Group, ExternalCode, IndepVarComp, Component
import numpy as np
import os
from getSamplePoints import getSamplePoints


class DakotaAEP(ExternalCode):
    """Use Dakota to estimate the AEP based on weighted power production."""

    def __init__(self, nDirections=10, dakotaFileName='dakotaAEP.in'):
        super(DakotaAEP, self).__init__()

        # set finite difference options (fd used for testing only)
        # self.fd_options['force_fd'] = True
        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-5
        self.fd_options['step_type'] = 'relative'

        # define inputs
        self.add_param('dirPowers', np.zeros(nDirections), units ='kW',
                       desc = 'vector containing the power production at each wind direction ccw from north') #TODO changed to dirPowers
        self.add_param('weights', np.zeros(nDirections),
                       desc = 'vector containing the weights for integration.') 
        self.add_param('windFrequencies', np.zeros(nDirections),
                       desc = 'vector containing the frequency from the probability density function.') #TODO changed to windFrequenceies

        # define output
        self.add_output('AEP', val=0.0, units='kWh', desc='total annual energy output of wind farm')
        self.add_output('Var_energy', val=0.0, units='kWh**2', desc='Variance of energy output of wind farm')

        # File in which the external code is implemented
        pythonfile = 'getDakotaAEP.py'
        self.options['command'] = ['python', pythonfile, dakotaFileName]

    def solve_nonlinear(self, params, unknowns, resids):

        # Generate the file with the power vector for Dakota
        power = params['dirPowers'] #TODO CHANGED
        rho = params['windFrequencies'] #TODO CHANGED
        power = power*rho
        np.savetxt('powerInput.txt', power, header='power')

        # parent solve_nonlinear function actually runs the external code
        super(DakotaAEP, self).solve_nonlinear(params,unknowns,resids)

        os.remove('powerInput.txt')

        # Read in the calculated AEP
        # number of hours in a year
        hours = 8760.0
        unknowns['AEP'] = np.loadtxt('AEP.txt')*hours
        unknowns['Var_energy'] = np.loadtxt('Var.txt')*hours


        print 'In DakotaAEP'

    def linearize(self, params, unknowns, resids):

        J = linearize_function(params)
        return J


class SimpleAEP(Component):
    """Use simple integration to estimate the AEP based on weighted power production."""

    def __init__(self, nDirections=10):

        super(SimpleAEP, self).__init__()

        # set finite difference options (fd used for testing only)
        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-5
        self.fd_options['step_type'] = 'relative'

        # define inputs
        self.add_param('dirPowers', np.zeros(nDirections), units ='kW',
                       desc = 'vector containing the power production at each wind direction ccw from north') #TODO CHANGED
        self.add_param('weights', np.zeros(nDirections),
                       desc = 'vector containing the weights for integration.')
        self.add_param('windFrequencies', np.zeros(nDirections),
                       desc = 'vector containing the frequency from the probability density function.') #TODO CHANGED

        # define output
        self.add_output('AEP', val=0.0, units='kWh', desc='total annual energy output of wind farm')
        self.add_output('std_energy', val=0.0, units='kWh', desc='standard deviation of energy output of wind farm')


    def solve_nonlinear(self, params, unknowns, resids):

        power = params['dirPowers'] #TODO CHANGED
        rho = params['windFrequencies'] #TODO CHANGED
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

        print 'In SimpleAEP'

    def linearize(self, params, unknowns, resids):

        J = linearize_function(params)
        return J


def linearize_function(params):

    power = params['dirPowers'] #TODO CHANGED
    weight = params['weights'] # The weights of the integration points
    rho = params['windFrequencies'] #TODO CHANGED
    # number of hours in a year
    hours = 8760.0
    dAEP_dpower = weight*rho*hours
    dAEP_dweight = power*rho*hours
    dAEP_drho = power*weight*hours

    J = {}
    J[('AEP', 'dirPowers')] = np.array([dAEP_dpower])
    J[('AEP', 'weights')] = np.array([dAEP_dweight])
    J[('AEP', 'windFrequencies')] = np.array([dAEP_drho])

    return J


if __name__ == "__main__":

    from wind_pdfs import wind_direction_pdf
    dakotaFileName = 'dakotaAEPdirection.in'
    # winddirections, weights = getSamplePoints(dakotaFileName)
    winddirections = np.linspace(0,360,10)
    weights = np.ones(len(winddirections))
    #f = wind_direction_pdf()
    rho = wind_direction_pdf(winddirections)
    prob = Problem(root=Group())
    prob.root.add('p', IndepVarComp('power', np.random.rand(10)))
    prob.root.add('w', IndepVarComp('weight', weights))
    prob.root.add('rho', IndepVarComp('frequency', rho))
    # prob.root.add('DakotaAEP', DakotaAEP(dakotaFileName=dakotaFileName))
    prob.root.add('DakotaAEP', SimpleAEP())
    prob.root.connect('p.power', 'DakotaAEP.dirPowers')
    prob.root.connect('w.weight', 'DakotaAEP.weights')
    prob.root.connect('rho.frequency', 'DakotaAEP.windFrequencies')
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
