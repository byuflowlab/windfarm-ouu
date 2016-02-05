from openmdao.api import Problem, Group, ExternalCode, IndepVarComp
import numpy as np


class DakotaAEP(ExternalCode):
    def __init__(self, nDirections=10):
        super(DakotaAEP, self).__init__()

        self.add_param('power_directions', np.zeros(nDirections), units ='kW',
                       desc = 'vector containing the power production at each wind direction ccw from north')
        self.add_param('windrose_frequencies', np.zeros(nDirections),
                       desc = 'vector containing the frequency of each wind speed at each direction')

        self.add_output('AEP', val=0.0, units='kWh', desc='total annual energy output of wind farm')

        # File in which the external code is implemented
        pythonfile = 'getDakotaAEP.py'
        self.options['command'] = ['python', pythonfile]


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

        # Question what if the optimizer only wants the function value.

        # The linearize returns but the solve_nonlinear doesn't
        # read gradient
        J = np.zeros(9)
        return J


if __name__ == "__main__":
    prob = Problem(root=Group())
    prob.root.add('p', IndepVarComp('power', np.random.rand(10)))
    prob.root.add('DakotaAEP', DakotaAEP())
    prob.root.connect('p.power', 'DakotaAEP.power_directions')
    prob.setup()
    prob.run()
    print 'AEP = ', (prob.root.DakotaAEP.unknowns['AEP'])
    print 'power directions = ', (prob.root.DakotaAEP.params['power_directions'])
