
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
import json
from openmdao.api import Problem
from florisse.floris import AEPGroupFLORIS
from dakotaGroups import dakotaGroupAEP
from getSamplePoints import getSamplePoints


def problem_set_up(windspeed, winddirection, weight, dakotaInputFile):
    """Set up wind farm problem.

    Args:
        windspeed (np.array): wind speeds
        winddirection (np.array): wind directions
        weight (np.array): weight associated with wind speed, wind direction pairs.
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
    # prob = Problem(AEPGroupFLORIS(nTurbines=nTurbs, nDirections=windDirections.size))
    prob = Problem(dakotaGroupAEP(nTurbines=nTurbs, nDirections=winddirection.size, dakotaFileName=dakotaInputFile))

    # initialize problem
    prob.setup(check=False)

    # assign initial values to variables
    prob['windSpeeds'] = windspeed
    prob['windDirections'] = winddirection
    prob['weights'] = weight
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generator_efficiency'] = generator_efficiency
    prob['air_density'] = air_density
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    for direction_id in range(0, winddirection.size):
        prob['yaw%i' % direction_id] = yaw

    # set options
    # prob['floris_params:FLORISoriginal'] = True
    # prob['floris_params:CPcorrected'] = False
    # prob['floris_params:CTcorrected'] = False

    return prob


def updateDakotaFile(dakotaFilename, quadrature_points):
    """Rewrite number of quadrature points in Dakota file."""

    filein = dakotaFilename
    fileout = dakotaFilename + '.tmp'
    fr = open(filein, 'r')
    fw = open(fileout, 'w')
    for line in fr:
        if 'quadrature_order' in line:
            towrite = 'quadrature_order  ' + str(quadrature_points) + '\n'
            fw.write(towrite)
        else:
            fw.write(line)
    fr.close()
    fw.close()
    shutil.move(fileout, filein)

if __name__ == "__main__":

    AEP = []
    samples = []
    for n in range(1,101,1):

        # n = 10  # Number of points
        dakotaFileName = 'dakotaAEPdirection.in'
        # dakotaFileName = 'dakotaAEPspeed.in'

        # Update dakota file with desired number of sample points
        updateDakotaFile(dakotaFileName, n)

        # run Dakota file to get the points locations (also weights)
        points, weights = getSamplePoints(dakotaFileName)       # The sample points could return 1 or 2D.
        # print 'Wind directions at which power is evaluated (deg)'
        print 'Wind speeds at which power is evaluated'
        a = 0
        b = 360
        points = (a+b)/2 + (b-a)/2*points
        for point in points:
            print '\t', point

        windspeed = np.ones(weights.size)*8
        winddirections = points

        # windspeed = points
        # winddirections = np.ones(weights.size)*225

        # Set up problem, define the turbine locations and all that stuff, pass it the wind direction x
        prob = problem_set_up(windspeed, winddirections, weights, dakotaFileName)

        # Run problem
        # get AEP, maybe variance as well.
        # Put the whole process in a loop.

        prob.run()

        # print the results
        hours = 8760.0 # number of hours in a year
        AEP_data = prob['AEP'] * hours
        print AEP_data
        AEP.append(AEP_data)
        samples.append(n)
        # os.remove('powerInput.txt')  # Because it gets read in in sample points...
        shutil.move('powerInput.txt', 'powerInput2.txt')


    power = np.loadtxt('powerInput2.txt')
    obj = {'AEP': AEP, 'samples': samples, 'points': points.tolist(), 'power': power.tolist()}
    jsonfile = open('record.json','w')
    json.dump(obj, jsonfile, indent=2)
    jsonfile.close()
    # jsonfile = open('record.json','r')
    # a = json.load(jsonfile)
    # print a
    # print type(a)
    # print a.keys()
    # print json.dumps(a)

    # fig, ax = plt.subplots()
    # ax.plot(samples,AEP)
    # ax.set_xlabel('Number of Wind Directions')
    # ax.set_ylabel('AEP')
    # ax.set_title('AEP as a function of the Number of Wind Directions')
    # plt.savefig('convergence.pdf')
    # plt.show()


