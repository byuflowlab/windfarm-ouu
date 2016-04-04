
import numpy as np
from openmdao.api import Problem
from AEPGroups import AEPGroup


def problem_set_up(windspeeds, winddirections, method='rect', dakotaInputFile=''):
    """Set up wind farm problem.

    Args:
        windspeeds (np.array): wind speeds vector
        winddirections (np.array): wind directions vector
        method (string): UQ method to use
        dakotaInputFile (string): filename of dakota input file

    Returns:
        prob (openMDAO problem class): The set up wind farm.

    """

    # define the size of farm, turbine size and operating conditions
    nRows = 10   # number of rows and columns in grid
    spacing = 5  # turbine grid spacing in diameters
    rotor_diameter = 126.4  # (m)
    air_density = 1.1716    # kg/m^3

    ### Set up the farm ###

    # Set up position of each turbine

    # Grid farm
    points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineX = np.ndarray.flatten(xpoints)
    turbineY = np.ndarray.flatten(ypoints)

    # Random farm
    # np.random.seed(101)
    # turbineX = np.random.rand(100)*spacing*(nRows-1)*rotor_diameter
    # turbineY = np.random.rand(100)*spacing*(nRows-1)*rotor_diameter

    # initialize arrays for each turbine properties
    nTurbs = turbineX.size
    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generator_efficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    # define initial values
    for turbI in range(nTurbs):
        rotorDiameter[turbI] = rotor_diameter
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generator_efficiency[turbI] = 0.944
        yaw[turbI] = 0.     # deg.

    # initialize problem
    prob = Problem(AEPGroup(nTurbines=nTurbs, nDirections=winddirections.size, method=method, dakotaFileName=dakotaInputFile))

    # initialize problem
    prob.setup(check=False)

    # assign initial values to variables
    prob['windSpeeds'] = windspeeds
    prob['windDirections'] = winddirections
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generator_efficiency'] = generator_efficiency
    prob['air_density'] = air_density
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    for direction_id in range(0, winddirections.size):
        prob['yaw%i' % direction_id] = yaw

    return prob
