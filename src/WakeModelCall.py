
import numpy as np
from openmdao.api import Problem
from WakeModelGroup import WakeModelGroup
from wakeexchange.floris import floris_wrapper, add_floris_params_IndepVarComps


def getPower(turbineX, turbineY, windDirections, windSpeeds, windWeights, gradient, analytic_gradient, wake_model, IndepVarFunc):
    """Calls the wake model, which is an openMDAO problem

    returns: The power for each wind direction and wind speed pair.
             The gradient of the power for each wind direction and wind speed pair
                with respect to the turbineX and turbineY locations.
    """

    # define turbine size
    rotor_diameter = 126.4  # (m)
    # Define flow properties
    air_density = 1.1716    # kg/m^3

    # initialize input variable arrays
    nTurbs = turbineX.size
    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = rotor_diameter      # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 0.944
        yaw[turbI] = 0.     # deg.

    # initialize problem
    n = windDirections.size
    prob = Problem(WakeModelGroup(nTurbines=nTurbs, nDirections=n, wake_model=wake_model, params_IdepVar_func=IndepVarFunc, analytic_gradient=analytic_gradient))
    prob.setup(check=False)

    # assign initial values to design variables
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    for direction_id in range(0, n):
        prob['yaw%i' % direction_id] = yaw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['windSpeeds'] = windSpeeds
    prob['air_density'] = air_density
    prob['windDirections'] = windDirections
    prob['windWeights'] = windWeights
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp

    prob.run()

    powers = prob['powerMUX.Array']

    # Compute the gradient of the power wrt to the turbine locations
    if gradient:
        J = prob.calc_gradient(['turbineX', 'turbineY'], ['powerMUX.Array'], return_format='dict')
        # To set finite difference options--step size, form, do so directly in the WakeModelGroup
        # J = prob.calc_gradient(['turbineX', 'turbineY'], ['powerMUX.Array'], return_format='dict', mode='fd')  # force fd mode with default step 1e-6 absolute
        JacobianX = J['powerMUX.Array']['turbineX']
        JacobianY = J['powerMUX.Array']['turbineY']

    else:
        JacobianX = np.array([None])
        JacobianY = np.array([None])

    return powers, JacobianX, JacobianY


if __name__ == "__main__":

    # define turbine locations in global reference frame
    turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
    turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])

    # Define flow properties
    wind_speed = 8.0        # m/s
    windDirections = np.linspace(0, 270, 4)
    n = windDirections.size
    windSpeeds = np.ones(n)*wind_speed
    windWeights = np.ones(n)/n

    wake_model = floris_wrapper
    IndepVarFunc = add_floris_params_IndepVarComps

    powers, JacobianX, JacobianY = getPower(turbineX, turbineY, windDirections, windSpeeds, windWeights,
                                            wake_model, IndepVarFunc)

    print 'powers = ', powers
    print 'dpower/dturbineX'
    print JacobianX
    print 'dpower/dturbineY'
    print JacobianY
    print JacobianY.shape

