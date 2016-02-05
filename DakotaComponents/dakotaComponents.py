from openmdao.api import Problem, Group, ExternalCode
import numpy as np

class SamplePoints(ExternalCode):
    def __init__(self):
        super(SamplePoints, self).__init__()

        # File in which the external code is implemented
        pythonfile = 'getSamplePoints.py'
        self.options['command'] = ['python', pythonfile]

        # Component outputs
        self.add_output('windDirections', shape=4)  # Shape needs to be determined (Sparse grid)


    def solve_nonlinear(self, params, unknowns, resids):

        # parent solve_nonlinear function actually runs the external code
        super(SamplePoints, self).solve_nonlinear(params,unknowns,resids)

        # postprocess the results
        dakotaTabular = 'dakota_tabular.dat'
        f = open(dakotaTabular,'r')
        f.readline()
        x = []
        for line in f:
            x.append(float(line.split()[2]))

        unknowns['windDirections'] = np.array(x)


class DakotaAEP(ExternalCode):
    def __init__(self, nDirections=1):
        super(DakotaAEP, self).__init__()

        self.add_param('power_directions', np.zeros(nDirections), units ='kW',
                       desc = 'vector containing the power production at each wind direction ccw from north')
        self.add_param('windrose_frequencies', np.zeros(nDirections),
                       desc = 'vector containing the frequency of each wind speed at each direction')

        self.add_output('AEP', val=0.0, units='kWh', desc='total annual energy output of wind farm')
        # Will need the x y locations.
        # Use these to overwrite the the dakota input file

        # File in which the external code is implemented
        pythonfile = 'getDakotaAEP.py'
        self.options['command'] = ['python', pythonfile]


    def solve_nonlinear(self, params, unknowns, resids):

        #preprocess() Set up x, y locations.

        # parent solve_nonlinear function actually runs the external code
        super(DakotaAEP, self).solve_nonlinear(params,unknowns,resids)

        # postprocess()
        unknowns['AEP'] = np.sum(params['power_directions'])
        #unknowns['AEPGradient'] = np.zeros(9)

    def linearize(self, params, unknowns, resids):

        # Question what if the optimizer only wants the function value.

        # The linearize returns but the solve_nonlinear doesn't
        # read gradient
        J = np.zeros(9)
        return J


if __name__ == "__main__":
    prob = Problem(root=Group())
    prob.root.add('samplePoints', SamplePoints())
    prob.setup()
    prob.run()
    print (prob.root.samplePoints.unknowns['windDirections'])
