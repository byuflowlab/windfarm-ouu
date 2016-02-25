from openmdao.api import Problem, Group
from florisse.floris import DirectionGroupFLORIS
import matplotlib.pyplot as plt
import random
import time
import numpy as np


if __name__ == "__main__":

    # define turbine locations in global reference frame
    rotor_diameter = 126.4  # (m)

    # Scaling grid case
    nRows = 10   # number of rows and columns in grid
    spacing = 5     # turbine grid spacing in diameters

    # Set up position arrays
    points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineX = np.ndarray.flatten(xpoints)
    turbineY = np.ndarray.flatten(ypoints)

    # turbineX = np.zeros(100)
    # turbineY = np.zeros(100)
    # for i in range(100):
    #    turbineX[i] = random.random()*6000
    #    turbineY[i]=  random.random()*6000

    plt.figure(1)
    plt.scatter(turbineX, turbineY)

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

    wind_direction = np.linspace(0,360,361)
    power = np.zeros(len(wind_direction))
    # wind_direction = 240    # deg (N = 0 deg., using direction FROM, as in met-mast data)
    # set up problem
    for i in range(0, len(wind_direction)):
        prob = Problem(root=Group())
        prob.root.add('FLORIS', DirectionGroupFLORIS(nTurbs, resolution=0), promotes=['*'])

        # initialize problem
        prob.setup()

        # assign values to turbine states
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['yaw0'] = yaw

        # assign values to constant inputs (not design variables)
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generator_efficiency'] = generator_efficiency
        prob['wind_speed'] = wind_speed
        prob['air_density'] = air_density
        prob['wind_direction'] = wind_direction[i]
        prob['Ct_in'] = Ct
        prob['Cp_in'] = Cp
        prob['floris_params:FLORISoriginal'] = False

        # run the problem
        tic = time.time()
        prob.run()
        toc = time.time()

        # print the results
        print 'wind farm power (kW): %s' % prob['power0']
        print i
        power[i] = prob['power0']

    np.savetxt("powerVSdirection_10x10_grid"+".txt", np.c_[wind_direction, power])
    plt.figure(2)
    plt.plot(wind_direction, power)
    plt.xlabel('wind_direction (degrees)')
    plt.ylabel('wind farm power (kW)')
    plt.title('Wind Farm Power As a Function of Wind Direction')
    plt.xlim([0, 360])
    plt.show()
