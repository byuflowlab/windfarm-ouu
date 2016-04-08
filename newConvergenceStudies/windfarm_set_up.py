
import numpy as np
from openmdao.api import Problem
from AEPGroups import AEPGroup


def problem_set_up(windspeed, winddirection, weight, rho, dakotaInputFile=''):
    """Set up wind farm problem.

    Args:
        windspeed (np.array): wind speeds vector
        winddirection (np.array): wind directions vector
        weight (np.array): weight associated with wind speed, wind direction pairs.
        rho (np.array): the value of the pdf associated with the wind speed or wind direction.
        dakotaInputFile (string): filename of dakota input file

    Returns:
        prob (openMDAO problem class): The set up wind farm.

    """

    # the wind directions and frequencies for the particular problem.

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
    prob = Problem(AEPGroup(nTurbines=nTurbs, nDirections=winddirection.size, dakotaFileName=dakotaInputFile))

    # initialize problem
    prob.setup(check=False)

    # assign initial values to variables
    prob['windSpeeds'] = windspeed
    prob['windDirections'] = winddirection
    prob['weights'] = weight
    prob['windFrequencies'] = rho #TODO CHANGED
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generator_efficiency #TODO CHANGED
    prob['air_density'] = air_density
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    for direction_id in range(0, winddirection.size):
        prob['yaw%i' % direction_id] = yaw

    return prob
