import numpy as np
import sys

from openmdao.api import Group, IndepVarComp
from statisticsComponents import DakotaStatistics, RectStatistics, ChaospyStatistics


class AEPGroup(Group):
    """
    Group containing all necessary components for wind plant AEP calculations
    """

    def __init__(self, nTurbines, nDirections=1, method_dict=None):

        super(AEPGroup, self).__init__()

        # Check derivatives
        # self.deriv_options['type'] = 'fd'
        # self.deriv_options['form'] = 'forward'
        # self.deriv_options['step_size'] = 1.0e-5
        # self.deriv_options['step_calc'] = 'relative'

        # providing default unit types
        direction_units = 'deg'
        wind_speed_units = 'm/s'
        length_units = 'm'

        # add necessary inputs for group
        self.add('dv0', IndepVarComp('windDirections', np.zeros(nDirections), units=direction_units), promotes=['*'])
        self.add('dv1', IndepVarComp('windSpeeds', np.zeros(nDirections), units=wind_speed_units), promotes=['*'])
        self.add('dv2', IndepVarComp('windWeights', np.zeros(nDirections)), promotes=['*'])

        self.add('dv3', IndepVarComp('turbineX', np.zeros(nTurbines), units=length_units), promotes=['*'])
        self.add('dv4', IndepVarComp('turbineY', np.zeros(nTurbines), units=length_units), promotes=['*'])

        method = method_dict['method']
        if method == 'dakota':
            self.add('AEPcomp', DakotaStatistics(nTurbines, nDirections, method_dict), promotes=['*'])
        elif method == 'chaospy':
            self.add('AEPcomp', ChaospyStatistics(nTurbines, nDirections, method_dict), promotes=['*'])
        elif method == 'rect':
            self.add('AEPcomp', RectStatistics(nTurbines, nDirections, method_dict), promotes=['*'])
        else:
            print "Specify one of these UQ methods = ['dakota', 'chaospy', 'rect']"
            sys.exit()

