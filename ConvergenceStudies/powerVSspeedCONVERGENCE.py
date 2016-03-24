from __future__ import print_function

from openmdao.api import Problem
from florisse.floris import AEPGroupFLORIS

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from windFreqFunctions import *

import cProfile


import sys

def wind_frequency_funcion():
    input_file = open("amalia_windrose_8.txt")
    wind_data = np.loadtxt(input_file)
    wind_data = np.append(wind_data, wind_data)

    length_data = np.linspace(0,144.01,len(wind_data))
    f = interp1d(length_data, wind_data)
    return f


def frequ(bins, f):
    offset = 12.
    offset = offset/5.
    bin_size = 72./bins
    dx = 0.01
    x1 = 0.
    x2 = x1+dx
    bin_location = bin_size
    frequency = np.zeros(bins)
    directions = np.array([])
    for i in range(0, bins):
        directions = np.append(directions, np.array([bin_location-bin_size, bin_location]))
        while x1 <= bin_location+offset:
            dfrequency = dx*(f(x1)+f(x2))/2
            frequency[i] += dfrequency
            x1 = x2
            x2 += dx
        bin_location += bin_size
    print('Directions: %s' % directions)
    return frequency


if __name__ == "__main__":

    ######################### for MPI functionality #########################
    from openmdao.core.mpi_wrap import MPI

    if MPI: # pragma: no cover
        # if you called this script with 'mpirun', then use the petsc data passing
        from openmdao.core.petsc_impl import PetscImpl as impl

    else:
        # if you didn't use 'mpirun', then use the numpy data passing
        from openmdao.api import BasicImpl as impl

    def mpi_print(prob, *args):
        """ helper function to only print on rank 0 """
        if prob.root.comm.rank == 0:
            print(*args)

    prob = Problem(impl=impl)

    #size = 4 # number of processors (and number of wind directions to run)

    #########################################################################
    # define turbine size
    rotor_diameter = 126.4  # (m)

    # define turbine locations in global reference frame
    # original example case
    # turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])   # m
    # turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])   # m

    # Scaling grid case
    nRows = 10   # number of rows and columns in grid
    spacing = 5     # turbine grid spacing in diameters

    # Set up position arrays
    points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineX = np.ndarray.flatten(xpoints)
    turbineY = np.ndarray.flatten(ypoints)

    # initialize input variable arrays
    nTurbs = turbineX.size
    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generator_efficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)
    minSpacing = 2.                         # number of rotor diameters

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = rotor_diameter      # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generator_efficiency[turbI] = 0.944
        yaw[turbI] = 0.     # deg.

    # wind_speed = 8.0        # m/s
    air_density = 1.1716    # kg/m^3

    AEP_data = np.array([])
    # ERROR = np.array([])
    number_speeds = np.array([])
    max_min = np.array([])
    max_speed = 30.

    for speeds in range(1, 500):
        # Define flow properties
        windDirections = np.ones(speeds)*225.
        windFrequencies = speed_frequ(speeds)
        wind_speeds = np.linspace(0, max_speed, speeds)

        # initialize problem
        prob = Problem(impl=impl, root=AEPGroupFLORIS(nTurbines=nTurbs, nDirections=windDirections.size, resolution=0))

        # initialize problem
        prob.setup(check=False)

        # time.sleep(10)
        # assign initial values to design variables
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generator_efficiency'] = generator_efficiency
        prob['windSpeeds'] = wind_speeds
        prob['air_density'] = air_density
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        for direction_id in range(0, windDirections.size):
            prob['yaw%i' % direction_id] = yaw

        # assign values to constant inputs (not design variables)
        prob['windDirections'] = windDirections
        prob['windrose_frequencies'] = windFrequencies
        prob['Ct_in'] = Ct

        prob['Cp_in'] = Cp

        # set options
        # prob['floris_params:FLORISoriginal'] = True
        # prob['floris_params:CPcorrected'] = False
        # prob['floris_params:CTcorrected'] = False

        # run the problem
        mpi_print(prob, 'start FLORIS run')
        tic = time.time()
        # cProfile.run('prob.run()')
        prob.run()
        toc = time.time()

        # print the results
        AEP_data = np.append(AEP_data, prob['AEP'])
        number_speeds = np.append(number_speeds, speeds)
        # if bins == 1:
        #     temp_error = 100
        # else:
        #     temp_error = abs((AEP_data[bins-1]-AEP_data[bins-2])/AEP_data[bins-1])*100.

        # ERROR = np.append(ERROR, temp_error)


        if speeds >= 30:
            max_min = np.append(max_min, prob['AEP'])

        mpi_print(prob,  'speeds: %s' % speeds)
        np.savetxt("weighted_convergence_AEP_VS_speeds.txt", np.c_[number_speeds, AEP_data])

    # plt.figure(1)
    # plt.plot(number_bins, AEP_data)
    # plt.ylim = ([0, 1.45e9])
    # plt.xlabel('Number of Wind Directions')
    # plt.ylabel('AEP')
    # plt.title('AEP as a function of the Number of Wind Directions (12 degree offset)')
    # # plt.savefig('AEP_bin_convergence_smoothed' + 'jpg')
    #
    # error = np.zeros(np.size(AEP_data))
    # error[0] = AEP_data[0]
    # for i in range(1, np.size(error)):
    #     error[i] = np.abs(AEP_data[i]-AEP_data[i-1])
    #
    # plt.figure(2)
    # plt.plot(number_bins, error)
    # plt.title('Error as a function of the Number of Wind Directions (12 degree offset)')
    # plt.xlabel('Number of Wind Directions')
    # plt.ylabel('Difference from previous AEP')
    # plt.show()
    #
    # maximum = np.max(max_min)
    # minimum = np.min(max_min)
    # average = np.mean(max_min)
    # max_error = abs(np.max([maximum, minimum])-average)/average
    # max_error_percent = max_error*100
    # mpi_print(prob, 'The maximum error after at least 30 bins is ', max_error_percent)
    #
    # """xbounds = [min(turbineX), min(turbineX), max(turbineX), max(turbineX), min(turbineX)]
    # ybounds = [min(turbineY), max(turbineY), max(turbineY), min(turbineY), min(turbineX)]
    #
    # plt.figure()
    # plt.plot(turbineX, turbineY, 'ok', label='Original')
    # plt.plot(prob['turbineX'], prob['turbineY'], 'og', label='Optimized')
    # plt.plot(xbounds, ybounds, ':k')
    # for i in range(0, nTurbs):
    #     plt.plot([turbineX[i], prob['turbineX'][i]], [turbineY[i], prob['turbineY'][i]], '--k')
    # plt.legend()
    # plt.xlabel('Turbine X Position (m)')
    # plt.ylabel('Turbine Y Position (m)')
    # plt.show()"""
