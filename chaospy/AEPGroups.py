import numpy as np
import sys

from openmdao.api import Group, IndepVarComp, ParallelGroup
from statisticsComponents import *

from florisse.GeneralWindFarmComponents import MUX, WindFarmAEP, DeMUX
from florisse.floris import DirectionGroupFLORIS


class AEPGroup(Group):
    """
    Group containing all necessary components for wind plant AEP calculations using the FLORIS model
    """

    def __init__(self, nTurbines, resolution=0, nDirections=1, use_rotor_components=False, datasize=0, method='', method_dict=None):

        super(AEPGroup, self).__init__()

        # providing default unit types for general MUX/DeMUX components
        power_units = 'kW'
        direction_units = 'deg'
        wind_speed_units = 'm/s'
        turbine_units = 'm'

        # add necessary inputs for group
        self.add('p1', IndepVarComp('windDirections', np.zeros(nDirections), units=direction_units), promotes=['*'])
        self.add('p2', IndepVarComp('windSpeeds', np.zeros(nDirections), units=wind_speed_units), promotes=['*'])

        self.add('p3', IndepVarComp('turbineX', np.zeros(nTurbines), units=turbine_units), promotes=['*'])
        self.add('p4', IndepVarComp('turbineY', np.zeros(nTurbines), units=turbine_units), promotes=['*'])

        # add vars to be seen by MPI and gradient calculations
        self.add('p5', IndepVarComp('rotorDiameter', np.zeros(nTurbines), units=turbine_units), promotes=['*'])
        self.add('p6', IndepVarComp('axialInduction', np.zeros(nTurbines)), promotes=['*'])
        self.add('p7', IndepVarComp('generator_efficiency', np.zeros(nTurbines)), promotes=['*'])
        self.add('p8', IndepVarComp('air_density', val=1.1716, units='kg/(m*m*m)'), promotes=['*'])


        # add components and groups
        self.add('windDirectionsDeMUX', DeMUX(nDirections, units=direction_units))
        self.add('windSpeedsDeMUX', DeMUX(nDirections, units=wind_speed_units))

        pg = self.add('all_directions', ParallelGroup(), promotes=['*'])

        # Can probably simplify the below rotor components logic
        if not use_rotor_components:
            self.add('p9', IndepVarComp('Ct_in', np.zeros(nTurbines)), promotes=['*'])
            self.add('p10', IndepVarComp('Cp_in', np.zeros(nTurbines)), promotes=['*'])

        if use_rotor_components:
            for direction_id in range(0, nDirections):
                pg.add('direction_group%i' % direction_id,
                       DirectionGroupFLORIS(nTurbines=nTurbines, resolution=resolution, direction_id=direction_id,
                                            use_rotor_components=use_rotor_components, datasize=datasize),
                       promotes=['params:*', 'floris_params:*', 'air_density',
                                 'axialInduction', 'generator_efficiency', 'turbineX', 'turbineY', 'rotorDiameter',
                                 'velocitiesTurbines%i' % direction_id, 'wt_power%i' % direction_id, 'power%i' % direction_id])#, 'wakeCentersYT', 'wakeDiametersT'])
        else:
            for direction_id in range(0, nDirections):
                pg.add('direction_group%i' % direction_id,
                       DirectionGroupFLORIS(nTurbines=nTurbines, resolution=resolution, direction_id=direction_id,
                                            use_rotor_components=use_rotor_components, datasize=datasize),
                       promotes=['Ct_in', 'Cp_in', 'params:*', 'floris_params:*', 'air_density',
                                 'axialInduction', 'generator_efficiency', 'turbineX', 'turbineY', 'rotorDiameter',
                                 'velocitiesTurbines%i' % direction_id, 'wt_power%i' % direction_id, 'power%i' % direction_id])#, 'wakeCentersYT', 'wakeDiametersT'])

        for direction_id in range(0, nDirections):
            self.add('y%i' % direction_id, IndepVarComp('yaw%i' % direction_id, np.zeros(nTurbines), units=direction_units), promotes=['*'])

        # Specify how the energy statistics are computed
        self.add('powerMUX', MUX(nDirections, units=power_units))
        if method == 'dakota':
            self.add('AEPcomp', DakotaStatistics(nDirections, method_dict), promotes=['*'])
        elif method == 'chaospy':
            self.add('AEPcomp', ChaospyStatistics(nDirections, method_dict), promotes=['*'])
        elif method == 'rect':
            self.add('AEPcomp', RectStatistics(nDirections, method_dict), promotes=['*'])
        else:
            print "Specify one of these UQ methods = ['dakota', 'chaospy', 'rect']"
            sys.exit()

        # connect components
        self.connect('windDirections', 'windDirectionsDeMUX.Array')
        self.connect('windSpeeds', 'windSpeedsDeMUX.Array')
        for direction_id in range(0, nDirections):
            self.connect('windDirectionsDeMUX.output%i' % direction_id, 'direction_group%i.wind_direction' % direction_id)
            self.connect('yaw%i' % direction_id, 'direction_group%i.yaw%i' % (direction_id, direction_id))
            self.connect('power%i' % direction_id, 'powerMUX.input%i' % direction_id)
            self.connect('windSpeedsDeMUX.output%i' % direction_id, 'direction_group%i.wind_speed' % direction_id)
        self.connect('powerMUX.Array', 'power')
        # self.connect('power', 'powerMUX.Array')  # reversing the order doesn't work.



