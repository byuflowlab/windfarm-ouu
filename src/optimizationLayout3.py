from __future__ import print_function

from openmdao.api import Problem, pyOptSparseDriver
from OptimizationGroup import OptAEP
from florisse.GeneralWindFarmComponents import calculate_boundary

import time
import numpy as np
import matplotlib.pyplot as plt
import json
import windfarm_setup
import distributions
from statistics_convergence import get_args


if __name__ == "__main__":

    #########################################################################
    # method_dict = {}
    # keys of method_dict:
    #     'method' = 'dakota', 'rect' or 'chaospy'  # 'chaospy' needs updating
    #     'uncertain_var' = 'speed' or 'direction'
    #     'layout' = 'amalia', 'optimized', 'grid', 'random', 'lhs', 'test'
    #                   'layout1', 'layout2', 'layout3'
    #     'distribution' = a distribution object
    #     'dakota_filename' = 'dakotaInput.in', applicable for dakota method

    args = get_args()
    method_dict = vars(args)  # Start a dictionary with the arguments specified in the command line
    method_dict['method']           = 'dakota'
    method_dict['uncertain_var']    = 'direction'
    method_dict['layout']           = 'layout3'

    if method_dict['uncertain_var'] == 'speed':
        dist = distributions.getWeibull()
        method_dict['distribution'] = dist
    elif method_dict['uncertain_var'] == 'direction':
        dist = distributions.getWindRose()
        method_dict['distribution'] = dist
    else:
        raise ValueError('unknown uncertain_var option "%s", valid options "speed" or "direction".' %method_dict['uncertain_var'])

    method_dict['dakota_filename'] = 'dakotageneral.in'

    n = 5  # number of points, i.e., number of winddirections and windspeeds pairs

    points, weights = windfarm_setup.getPoints(method_dict, n)

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

    print('Locations at which power is evaluated')
    print('\twindspeed \t winddirection')
    for i in range(n):
        print(i+1, '\t', '%.2f' % windspeeds[i], '\t', '%.2f' % winddirections[i])

    # Turbines layout
    turbineX, turbineY = windfarm_setup.getLayout(method_dict['layout'])

    locations = np.column_stack((turbineX, turbineY))
    # generate boundary constraint
    # boundaryVertices, boundaryNormals = calculate_boundary(locations)
    #amalia boundaryVertices
    boundaryVertices = np.array([   [ 3710.176, 3569.028],
                                     [ 1683.694, 4889.77 ],
                                     [ 1124.143, 4869.606],
                                     [  297.419, 4390.711],
                                     [   20.164, 3911.816],
                                     [    0.   , 2948.985],
                                     [  216.763, 1497.177],
                                     [  972.913,   10.082],
                                     [ 1552.628,    0.   ],
                                     [ 2157.548,   20.164],
                                     [ 3135.502,  579.715],
                                     [ 3483.331, 1103.979],
                                     [ 3800.914, 1633.284],
                                     [ 3780.75 , 2611.238]])

    boundaryNormals = np.array([     [ 0.54601347, 0.8377764 ],
                                     [-0.03601266, 0.99935133],
                                     [-0.50124424, 0.86530585],
                                     [-0.86542629, 0.50103627],
                                     [-0.99978078, 0.02093782],
                                     [-0.98903688,-0.1476687 ],
                                     [-0.89138513,-0.45324668],
                                     [-0.01738867,-0.99984881],
                                     [ 0.03331483,-0.99944491],
                                     [ 0.49662068,-0.86796768],
                                     [ 0.8332809 ,-0.55284983],
                                     [ 0.85749293,-0.51449576],
                                     [ 0.99978751, 0.02061418],
                                     [ 0.99729632, 0.07348499]])


    nVertices = boundaryVertices.shape[0]
    print('boundary vertices', boundaryVertices)
    print('boundary normals', boundaryNormals)

    # turbine size and operating conditions
    rotor_diameter = 126.4  # (m)
    air_density = 1.1716    # kg/m^3

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
    prob = Problem(root=OptAEP(nTurbines=nTurbs, nDirections=n, minSpacing=minSpacing, use_rotor_components=False, differentiable=True, nVertices=nVertices, method_dict=method_dict))

    # set up optimizer
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.add_objective('obj', scaler=1E-8)  # the amalia has the scaler at 1e-5, originally 1E-8

    # set optimizer options
    prob.driver.opt_settings['Verify level'] = 3
    prob.driver.opt_settings['Print file'] = 'SNOPT_print_layout3_5_dakota.out'
    prob.driver.opt_settings['Summary file'] = 'SNOPT_summary_layout3_5_dakota.out'
    prob.driver.opt_settings['Major iterations limit'] = 1000
    prob.driver.opt_settings['Major optimality tolerance'] = 2E-6


    # select design variables
    prob.driver.add_desvar('turbineX', scaler=1.0)
    prob.driver.add_desvar('turbineY', scaler=1.0)
    # for direction_id in range(0, n):
    #     prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1.0)

    # add constraints
    # prob.driver.add_constraint('sc', lower=np.zeros(((nTurbs-1.)*nTurbs/2.)), scaler=1.0/rotor_diameter)
    prob.driver.add_constraint('sc', lower=np.zeros(((nTurbs-1.)*nTurbs/2.)), scaler=1.0)
    prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbs), scaler=1.0)

    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True
    tic = time.time()
    prob.setup(check=False)
    toc = time.time()

    # print the results
    print('FLORIS setup took %.03f sec.' % (toc-tic))

    # time.sleep(10)
    # assign initial values to design variables
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    for direction_id in range(0, n):
        prob['yaw%i' % direction_id] = yaw

    # assign values to constant inputs (not design variables)
    prob['windSpeeds'] = windspeeds
    prob['windDirections'] = winddirections
    prob['weights'] = weights
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['air_density'] = air_density
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp

    # provide values for the hull constraint
    prob['boundaryVertices'] = boundaryVertices
    prob['boundaryNormals'] = boundaryNormals

    # set options
    # prob['floris_params:FLORISoriginal'] = True
    # prob['floris_params:CPcorrected'] = False
    # prob['floris_params:CTcorrected'] = False

    # run the problem
    print(prob, 'start FLORIS run')
    tic = time.time()
    prob.run()
    toc = time.time()

    # print the results
    print('FLORIS Opt. calculation took %.03f sec.' % (toc-tic))

    #for direction_id in range(0, n):
    #    print('yaw%i (deg) = ' % direction_id, prob['yaw%i' % direction_id])
    # for direction_id in range(0, n):
        # mpi_print(prob,  'velocitiesTurbines%i (m/s) = ' % direction_id, prob['velocitiesTurbines%i' % direction_id])
    # for direction_id in range(0, n):
    #     mpi_print(prob,  'wt_power%i (kW) = ' % direction_id, prob['wt_power%i' % direction_id])

    print('turbine X positions in wind frame (m): %s' % prob['turbineX'])
    print('turbine Y positions in wind frame (m): %s' % prob['turbineY'])
    print('wind farm power in each direction (kW): %s' % prob['power'])
    print('AEP (kWh): %s' % prob['mean'])

    xbounds = [min(turbineX), min(turbineX), max(turbineX), max(turbineX), min(turbineX)]
    ybounds = [min(turbineY), max(turbineY), max(turbineY), min(turbineY), min(turbineX)]

    np.savetxt('layout_3_5_XY_dakota.txt', np.c_[prob['turbineX'], prob['turbineY']], header="turbineX, turbineY")

    # Save details of the simulation
    obj = {'mean': prob['mean']/1e6, 'std': prob['std']/1e6, 'samples': n, 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': prob['power'].tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
           'layout': method_dict['layout'], 'turbineX': turbineX.tolist(), 'turbineY': turbineY.tolist(),
           'turbineXopt': prob['turbineX'].tolist(), 'turbineYopt': prob['turbineY'].tolist()}
    jsonfile = open('layout_3_5_dakota.json','w')
    json.dump(obj, jsonfile, indent=2)
    jsonfile.close()


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
