from openmdao.api import Problem, Group, ExternalCode, IndepVarComp
import numpy as np
from getSamplePoints import getSamplePoints


class DakotaAEP(ExternalCode):
    def __init__(self, nDirections=10, dakotaFileName='dakotaAEP.in'):
        super(DakotaAEP, self).__init__()

        # set finite difference options (fd used for testing)
        # self.fd_options['force_fd'] = True
        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-4
        self.fd_options['step_type'] = 'relative'


        self.add_param('power_directions', np.zeros(nDirections), units ='kW',
                       desc = 'vector containing the power production at each wind direction ccw from north')
        self.add_param('weights', np.zeros(nDirections),
                       desc = 'vector containing the frequency of each wind speed at each direction')

        self.add_output('AEP', val=0.0, units='kWh', desc='total annual energy output of wind farm')

        # File in which the external code is implemented
        pythonfile = 'getDakotaAEP.py'
        self.options['command'] = ['python', pythonfile, dakotaFileName]


    def solve_nonlinear(self, params, unknowns, resids):

        # Generate the file with the power directions for Dakota
        power = params['power_directions']
        np.savetxt('powerInput.txt', power, header='power')

        # parent solve_nonlinear function actually runs the external code
        super(DakotaAEP, self).solve_nonlinear(params,unknowns,resids)

        # Read in the calculated AEP

        unknowns['AEP'] = np.loadtxt('AEP.txt')
        #unknowns['AEP'] = np.sum(params['power_directions'])


    def linearize(self, params, unknowns, resids):

        weight = params['weights'] # The weights of the integration points
        dAEP_dpower = weight

        J = {}
        J[('AEP', 'power_directions')] = np.array([dAEP_dpower])

        return J


if __name__ == "__main__":

    dakotaFileName = 'dakotaAEP.in'
    unused, weights = getSamplePoints(dakotaFileName)

    prob = Problem(root=Group())
    prob.root.add('p', IndepVarComp('power', np.random.rand(10)))
    prob.root.add('w', IndepVarComp('weight', weights))
    prob.root.add('DakotaAEP', DakotaAEP())
    prob.root.connect('p.power', 'DakotaAEP.power_directions')
    prob.root.connect('w.weight', 'DakotaAEP.weights')
    prob.setup()
    prob.run()
    print 'AEP = ', (prob.root.DakotaAEP.unknowns['AEP'])
    print 'power directions = ', (prob.root.DakotaAEP.params['power_directions'])
    print prob.root.DakotaAEP.params.keys()
    # The DakotaAEP.power_directions key is not recognized
    # J = prob.calc_gradient(['DakotaAEP.AEP'], ['DakotaAEP.power_directions'])
    # J = prob.calc_gradient(['DakotaAEP.AEP'], ['p.power'])
    # print 'power directions gradient = ', J

    # This check works
    data = prob.check_partial_derivatives()
