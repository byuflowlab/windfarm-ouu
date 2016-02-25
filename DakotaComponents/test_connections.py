import numpy as np

from openmdao.api import Group, Component, Problem, IndepVarComp, ParamComp, ParallelGroup, NLGaussSeidel, ScipyGMRES

import dakotaComponents

from florisse.GeneralWindFarmComponents import WindFrame, AdjustCtCpYaw, MUX, WindFarmAEP, DeMUX, CPCT_Interpolate_Gradients
from florisse.Parameters import FLORISParameters
from florisse.floris import DirectionGroupFLORIS
import _floris

from dakotaGroups import dakotaGroupAEP

if __name__ == '__main__':

    # define turbine locations in global reference frame
    turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
    turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])

    # initialize input variable arrays
    nTurbs = turbineX.size
    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generator_efficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = 126.4            # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generator_efficiency[turbI] = 0.944
        yaw[turbI] = 0.     # deg.

    # Define flow properties
    wind_speed = 8.0        # m/s
    air_density = 1.1716    # kg/m^3
    nDirections = 10
    windDirections = np.linspace(0, 360.-360./nDirections, nDirections)
    windSpeeds = np.linspace(3, 12, 10)
    windFrequencies = np.ones_like(windDirections)*1.0/nDirections  # deg (N = 0 deg., using direction FROM, as in met-mast data)

    # set up problem
    prob = Problem(dakotaGroupAEP(nTurbines=nTurbs, nDirections=nDirections))

    # initialize problem
    prob.setup(check=True)

    # assign values to turbine states
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    for direction_id in range(0, windDirections.size):
        prob['yaw%i' % direction_id]

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generator_efficiency'] = generator_efficiency
    prob['windSpeeds'] = windSpeeds
    prob['air_density'] = air_density
    prob['windDirections'] = windDirections
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp
    prob['floris_params:FLORISoriginal'] = False
    prob['windrose_frequencies'] = windFrequencies

    # run the problem
    print 'start FLORIS run'
    prob.run()

    # print the results
    print 'power_directions = ', prob['power_directions']
    print 'AEP = ', prob['AEP']




