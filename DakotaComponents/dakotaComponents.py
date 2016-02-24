from openmdao.api import Problem, Group, ExternalCode, IndepVarComp
import numpy as np


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
        self.add_param('windrose_frequencies', np.zeros(nDirections),
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

        power = params['power_directions']

        # Get the weights of the integration points
        # weights = [0.03366682575, 0.07745913394, 0.1283591748, 0.1342931983, 0.1567958649, 0.2109227504, 0.1628296759, 0.06054507962, 0.02454240261, 0.01058589391]
        dakotaTabular = 'dakota_quadrature_tabular.dat'
        f = open(dakotaTabular, 'r')
        f.readline()
        w = []
        for line in f:
            w.append(float(line.split()[1]))

        dAEP_dpower = np.ones(len(power))*np.array(w)

        J = {}
        J[('AEP', 'power_directions')] = np.array([dAEP_dpower])

        # Do we need this partial? This is not as straighforward as below
        # dAEP_dwindrose_frequencies = np.ones(ndirs)*power_directions*hours


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
    print prob.root.DakotaAEP.params.keys()
    # The DakotaAEP.power_directions key is not recognized
    # J = prob.calc_gradient(['DakotaAEP.AEP'], ['DakotaAEP.power_directions'])
    # J = prob.calc_gradient(['DakotaAEP.AEP'], ['p.power'])
    # print 'power directions gradient = ', J

    # This check works
    data = prob.check_partial_derivatives()
