from __future__ import print_function

from openmdao.api import Problem, pyOptSparseDriver
from OptimizationGroup import OptAEP
from florisse import config

import distributions
import quadrature_rules
import time
import numpy as np
import pylab as plt

import cProfile


import sys

if __name__ == "__main__":

    config.floris_single_component = True

    n = 10  # number of processors (and number of wind directions to run)

    #########################################################################

    method = 'rect'
    dist = distributions.getWindRose()

    method_dict = {}
    method_dict['distribution'] = dist

    points, unused = quadrature_rules.rectangle(n, method_dict['distribution'])

    # windSpeeds = points[0]
    # windDirections = np.ones(n)*225

    windSpeeds = np.ones(n)*8  # m/s
    windDirections = points[0]

    print('Locations at which power is evaluated')
    print('\twindSpeed \t windDirection')
    for i in range(n):
        print(i+1,'\t', '%.2f' %windSpeeds[i], '\t', '%.2f' %windDirections[i])


    # define the size of farm, turbine size and operating conditions
    nRows = 2  # 10   # number of rows and columns in grid
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

    # initialize input variable arrays
    nTurbs = turbineX.size
    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)
    minSpacing = 2.                         # number of rotor diameters

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = rotor_diameter      # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 0.944
        yaw[turbI] = 0.     # deg.

    # initialize problem
    prob = Problem(root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, minSpacing=minSpacing, use_rotor_components=False, datasize=0, differentiable=True, force_fd=False, nSamples=0, method=method, method_dict=method_dict))

    # set up optimizer
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.add_objective('obj', scaler=1E-8)

    # set optimizer options
    prob.driver.opt_settings['Verify level'] = 3
    prob.driver.opt_settings['Print file'] = 'SNOPT_print_exampleOptAEP.out'
    prob.driver.opt_settings['Summary file'] = 'SNOPT_summary_exampleOptAEP.out'
    prob.driver.opt_settings['Major iterations limit'] = 1000

    # select design variables
    prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs)*min(turbineX), upper=np.ones(nTurbs)*max(turbineX), scaler=1E-2)
    prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY), scaler=1E-2)
    for direction_id in range(0, windDirections.size):
        prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1E-1)

    # add constraints
    prob.driver.add_constraint('sc', lower=np.zeros(((nTurbs-1.)*nTurbs/2.)), scaler=1.0/rotor_diameter)

    tic = time.time()
    prob.setup(check=False)
    toc = time.time()

    # print the results
    print('FLORIS setup took %.03f sec.' % (toc-tic))

    # time.sleep(10)
    # assign initial values to design variables
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    for direction_id in range(0, windDirections.size):
        prob['yaw%i' % direction_id] = yaw

    # assign values to constant inputs (not design variables)
    prob['windSpeeds'] = windSpeeds
    prob['windDirections'] = windDirections
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['air_density'] = air_density
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp

    # set options
    # prob['floris_params:FLORISoriginal'] = True
    # prob['floris_params:CPcorrected'] = False
    # prob['floris_params:CTcorrected'] = False

    # run the problem
    print(prob, 'start FLORIS run')
    tic = time.time()
    # cProfile.run('prob.run()')
    prob.run()
    toc = time.time()

    # print the results
    print('FLORIS Opt. calculation took %.03f sec.' % (toc-tic))

    for direction_id in range(0, windDirections.size):
        print('yaw%i (deg) = ' % direction_id, prob['yaw%i' % direction_id])
    # for direction_id in range(0, windDirections.size):
        # mpi_print(prob,  'velocitiesTurbines%i (m/s) = ' % direction_id, prob['velocitiesTurbines%i' % direction_id])
    # for direction_id in range(0, windDirections.size):
    #     mpi_print(prob,  'wt_power%i (kW) = ' % direction_id, prob['wt_power%i' % direction_id])

    print('turbine X positions in wind frame (m): %s' % prob['turbineX'])
    print('turbine Y positions in wind frame (m): %s' % prob['turbineY'])
    print('wind farm power in each direction (kW): %s' % prob['dirPowers'])
    print('AEP (kWh): %s' % prob['mean'])

    xbounds = [min(turbineX), min(turbineX), max(turbineX), max(turbineX), min(turbineX)]
    ybounds = [min(turbineY), max(turbineY), max(turbineY), min(turbineY), min(turbineX)]

    plt.figure()
    plt.plot(turbineX, turbineY, 'ok', label='Original')
    plt.plot(prob['turbineX'], prob['turbineY'], 'og', label='Optimized')
    plt.plot(xbounds, ybounds, ':k')
    for i in range(0, nTurbs):
        plt.plot([turbineX[i], prob['turbineX'][i]], [turbineY[i], prob['turbineY'][i]], '--k')
    plt.legend()
    plt.xlabel('Turbine X Position (m)')
    plt.ylabel('Turbine Y Position (m)')
    plt.show()


