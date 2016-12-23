import numpy as np
import sys

from openmdao.api import Group, IndepVarComp
from statisticsComponents import DakotaStatistics, RectStatistics, ChaospyStatistics

from wakeexchange.floris import floris_wrapper, add_floris_params_IndepVarComps


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

        # providing default unit types
        power_units = 'kW'
        length_units = 'm'

        # add necessary inputs for group
        self.add('dv0', IndepVarComp('windWeights', np.zeros(nDirections)), promotes=['*'])

        self.add('dv1', IndepVarComp('turbineX', np.zeros(nTurbines), units=length_units), promotes=['*'])
        self.add('dv2', IndepVarComp('turbineY', np.zeros(nTurbines), units=length_units), promotes=['*'])

        self.add('dv3', IndepVarComp('Powers', np.zeros(nDirections), units=power_units), promotes=['*'])
        self.add('dv4', IndepVarComp('dpower_dturbX', np.zeros([nDirections, nTurbines]), units='kW/m'), promotes=['*'])
        self.add('dv5', IndepVarComp('dpower_dturbY', np.zeros([nDirections, nTurbines]), units='kW/m'), promotes=['*'])

        method = method_dict['method']
        if method == 'dakota':
            self.add('AEPcomp', DakotaStatistics(nTurbines, nDirections, method_dict), promotes=['*'])
        elif method == 'chaospy':
            self.add('AEPcomp', ChaospyStatistics(nDirections, method_dict), promotes=['*'])
        elif method == 'rect':
            self.add('AEPcomp', RectStatistics(nTurbines, nDirections, method_dict), promotes=['*'])
        else:
            print "Specify one of these UQ methods = ['dakota', 'chaospy', 'rect']"
            sys.exit()

