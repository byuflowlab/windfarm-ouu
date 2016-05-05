
import numpy as np
import chaospy as cp
# import matplotlib.pyplot as plt
from openmdao.api import Problem
from AEPGroups import AEPGroup


def problem_set_up(windspeeds, winddirections, method_dict=None):
    """Set up wind farm problem.

    Args:
        windspeeds (np.array): wind speeds vector
        winddirections (np.array): wind directions vector
        method_dict (dict): UQ method and parameters for the UQ method

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
    # points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    # xpoints, ypoints = np.meshgrid(points, points)
    # turbineX = np.ndarray.flatten(xpoints)
    # turbineY = np.ndarray.flatten(ypoints)

    # Random farm
    np.random.seed(101)
    turbineX = np.random.rand(100)*spacing*(nRows-1)*rotor_diameter
    turbineY = np.random.rand(100)*spacing*(nRows-1)*rotor_diameter

    # Latin Hypercube farm
    # dist = cp.Iid(cp.Uniform(0, spacing*(nRows-1)*rotor_diameter), 2)
    # x = dist.sample(100, 'L')
    # turbineX = x[0]
    # turbineY = x[1]

    # Amalia wind farm
    # locations = np.genfromtxt('../WindFarms/layout_amalia.txt', delimiter=' ')
    # turbineX = locations[:,0]
    # turbineY = locations[:,1]

    # Amalia optimized with 4 wind dir
    # locations = np.genfromtxt('../WindFarms/layout_amalia_optimized4dir.csv', delimiter=',')
    # turbineX = locations[:,0]
    # turbineY = locations[:,1]

    # Amalia optimized Jared
    # locations = np.genfromtxt('../WindFarms/AmaliaOptimizedXY.txt', delimiter=' ')
    # turbineX = locations[:,0]
    # turbineY = locations[:,1]

    # For printing the location as an array
    # print turbineX
    # print turbineY
    # a = '['
    # for x in turbineX:
    #     a = a + '%.0f' % x + ', '
    # print 'turbineX', a
    # a = '['
    # for y in turbineY:
    #     a = a + '%.0f' % y + ', '
    # print 'turbineY', a


    # plt.figure()
    # plt.scatter(turbineX, turbineY)
    # plt.show()

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
    prob = Problem(AEPGroup(nTurbines=nTurbs, nDirections=winddirections.size,
                            method_dict=method_dict))

    # initialize problem
    prob.setup(check=False)

    # assign initial values to variables
    prob['windSpeeds'] = windspeeds
    prob['windDirections'] = winddirections
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generator_efficiency
    prob['air_density'] = air_density
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    for direction_id in range(0, winddirections.size):
        prob['yaw%i' % direction_id] = yaw

    return prob
