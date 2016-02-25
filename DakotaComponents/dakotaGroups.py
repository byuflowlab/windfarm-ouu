import numpy as np

from openmdao.api import Group, Component, Problem, IndepVarComp, ParamComp, ParallelGroup, NLGaussSeidel, ScipyGMRES

import dakotaComponents

from florisse.GeneralWindFarmComponents import WindFrame, AdjustCtCpYaw, MUX, WindFarmAEP, DeMUX, CPCT_Interpolate_Gradients
from florisse.Parameters import FLORISParameters
from florisse.floris import DirectionGroupFLORIS
import _floris


class dakotaGroupAEP(Group):
    """
    Group containing all necessary components for wind plant AEP calculations using the FLORIS model
    """

    def __init__(self, nTurbines, resolution=0, nDirections=1, use_rotor_components=False, datasize=0, dakotaFileName='dakotaAEP.in'):

        super(dakotaGroupAEP, self).__init__()

        # add components and groups
        self.add('windDirectionsDeMUX', DeMUX(nDirections))
        self.add('windSpeedsDeMUX', DeMUX(nDirections))

        pg = self.add('all_directions', ParallelGroup(), promotes=['*'])
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

        self.add('powerMUX', MUX(nDirections))
        self.add('AEPcomp', dakotaComponents.DakotaAEP(nDirections, dakotaFileName), promotes=['*'])

        # add necessary inputs for group
        self.add('p1', IndepVarComp('windDirections', np.zeros(nDirections)), promotes=['*'])
        self.add('p2', IndepVarComp('turbineX', np.zeros(nTurbines)), promotes=['*'])
        self.add('p3', IndepVarComp('turbineY', np.zeros(nTurbines)), promotes=['*'])

        # add vars to be seen by MPI and gradient calculations
        self.add('p5', IndepVarComp('rotorDiameter', np.zeros(nTurbines)), promotes=['*'])
        self.add('p6', IndepVarComp('axialInduction', np.zeros(nTurbines)), promotes=['*'])
        self.add('p7', IndepVarComp('generator_efficiency', np.zeros(nTurbines)), promotes=['*'])
        # self.add('p8', IndepVarComp('windSpeeds', val=8.0), promotes=['*'])
        self.add('p8', IndepVarComp('windSpeeds', np.zeros(nDirections), units='m/s'), promotes=['*'])
        self.add('p9', IndepVarComp('air_density', val=1.1716), promotes=['*'])
        self.add('p11', IndepVarComp('windrose_frequencies', np.zeros(nDirections)), promotes=['*'])

        if not use_rotor_components:
            self.add('p12', IndepVarComp('Ct_in', np.zeros(nTurbines)), promotes=['*'])
            self.add('p13', IndepVarComp('Cp_in', np.zeros(nTurbines)), promotes=['*'])


        for direction_id in range(0, nDirections):
            self.add('y%i' % direction_id, IndepVarComp('yaw%i' % direction_id, np.zeros(nTurbines)), promotes=['*'])

        # connect components
        self.connect('windDirections', 'windDirectionsDeMUX.Array')
        self.connect('windSpeeds', 'windSpeedsDeMUX.Array')
        for direction_id in range(0, nDirections):
            self.connect('windDirectionsDeMUX.output%i' % direction_id, 'direction_group%i.wind_direction' % direction_id)
            self.connect('yaw%i' % direction_id, 'direction_group%i.yaw%i' % (direction_id, direction_id))
            self.connect('power%i' % direction_id, 'powerMUX.input%i' % direction_id)
            self.connect('windSpeedsDeMUX.output%i' % direction_id, 'direction_group%i.wind_speed' % direction_id)
        self.connect('powerMUX.Array', 'power_directions')

