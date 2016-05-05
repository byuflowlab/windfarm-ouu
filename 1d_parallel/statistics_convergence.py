
import numpy as np
# import matplotlib.pyplot as plt
import chaospy as cp
import json
from openmdao.api import Problem
from AEPGroups import AEPGroup
from getSamplePoints import getSamplePoints
from dakotaInterface import updateDakotaFile, updateDakotaFile2
import distributions
import quadrature_rules


def getPoints(method_dict, n):

    method = method_dict['method']

    if method == 'dakota':
        # Update dakota file with desired number of sample points
        updateDakotaFile(method_dict['dakota_filename'], n)
        # run Dakota file to get the points locations
        points = getSamplePoints(method_dict['dakota_filename'])

        # If direction
        # Update the points to correct range
        a = 140
        b = 470
        points = ((b+a)/2. + (b-a)*points)%360
        return points

    if method == 'rect':

        if method_dict['uncertain_var'] == 'speed':
            dist = distributions.getWeibull()
            method_dict['distribution'] = dist
        elif method_dict['uncertain_var'] == 'direction':
            dist = distributions.getWindRose()
            method_dict['distribution'] = dist
        else:
            raise ValueError('unknown uncertain_var option "%s", valid options "speed" or "direction".' %method_dict['uncertain_var'])

        points, unused = quadrature_rules.rectangle(n, method_dict['distribution'])
        return points[0]


def run():
    """
    method_dict = {}
    keys of method_dict:
        'method' = 'dakota', 'rect' or 'chaospy'  # 'chaospy needs updating
        'uncertain_var' = 'speed' or 'direction'
        'dakota_filename' = 'dakotaInput.in', applicable for dakota method
        'distribution' = a distribution applicable for rect and chaospy methods, it gets set in getPoints()
    Returns:
        Writes a json file 'record.json' with the run information.
    """

    method_dict = {}
    method_dict['method'] = 'rect'
    method_dict['uncertain_var'] = 'direction'

    # method_dict['dakota_filename'] = 'dakotaAEPspeed.in'
    # method_dict['dakota_filename'] = 'dakotaAEPdirection.in'
    method_dict['dakota_filename'] = 'dakotadirectionsmooth.in'

    mean = []
    std = []
    samples = []

    for n in range(20,21,1):

        points = getPoints(method_dict, n)

        if method_dict['uncertain_var'] == 'speed':
            # For wind speed
            windspeeds = points
            winddirections = np.ones(n)*225
        elif method_dict['uncertain_var'] == 'direction':
            # For wind direction
            windspeeds = np.ones(n)*8
            winddirections = points
        else:
            raise ValueError('unknown uncertain_var option "%s", valid options "speed" or "direction".' %method_dict['uncertain_var'])


        print 'Locations at which power is evaluated'
        print '\twindspeed \t winddirection'
        for i in range(n):
            print i+1, '\t', '%.2f' % windspeeds[i], '\t', '%.2f' % winddirections[i]

        # Set up problem, define the turbine locations and all that stuff
        prob = problem_set_up(windspeeds, winddirections, method_dict)

        prob.run()

        # print the results
        mean_data = prob['mean']
        std_data = prob['std']
        print 'mean = ', mean_data/1e6, ' GWhrs'
        print 'std = ', std_data/1e6, ' GWhrs'
        mean.append(mean_data/1e6)
        std.append(std_data/1e6)
        samples.append(n)


    # Save a record of the run
    power = prob['power']

    obj = {'mean': mean, 'std': std, 'samples': samples, 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': power.tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var']}
    jsonfile = open('record.json','w')
    json.dump(obj, jsonfile, indent=2)
    jsonfile.close()


def plot():
    jsonfile = open('record.json','r')
    a = json.load(jsonfile)
    print a
    print type(a)
    print a.keys()
    print json.dumps(a, indent=2)

    # fig, ax = plt.subplots()
    # ax.plot(windspeeds, power)
    # ax.set_xlabel('wind speed (m/s)')
    # ax.set_ylabel('power')
    #
    # fig, ax = plt.subplots()
    # ax.plot(samples,mean)
    # ax.set_xlabel('Number of Wind Directions')
    # ax.set_ylabel('mean annual energy production')
    # ax.set_title('Mean annual energy as a function of the Number of Wind Directions')
    #
    # plt.show()


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

if __name__ == "__main__":
    run()
    # plot()


