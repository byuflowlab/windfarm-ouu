from __future__ import print_function

from openmdao.api import Problem, pyOptSparseDriver
from OptimizationGroup import OptAEP
from florisse import config
from florisse.GeneralWindFarmComponents import calculate_boundary

from getSamplePoints import getSamplePoints
from dakotaInterface import updateDakotaFile
import distributions

import time
import numpy as np
import pylab as plt
import chaospy as cp

import cProfile


import sys

def modifyx(x, A=110, B=140, C=225, r=360):

    # Modify x, to start from the max probability location
    x = (C+x)%r
    y = []
    for xi in x:
        if A<C:
            if xi > A and xi < C:
                xi = (xi + B-A)%r
            y.append(xi)
        else:
            if xi > A:
                xi = (xi + B-A)%r
            y.append(xi)
    return np.array(y)


def getWeights(x, dx, dist):
    # Logic to get the weights from integrating the pdf between the bins
    w = []
    for xi in x:
        xleft = xi-dx/2.
        xright = xi+dx/2.
        if xright > 360.0:
            w.append(1 - dist._cdf(xleft) + dist._cdf(xright-360))
        elif xleft < 0.0:
            w.append(dist._cdf(xright) + (1 - dist._cdf(360+xleft)))
        else:
            w.append(dist._cdf(xright) - dist._cdf(xleft))
        # print xi+dx/2., xi-dx/2.
    w = np.array(w).flatten()
    # print w  # all weights should be positive
    # print np.sum(w)   # this should sum to 1
    return w


def getPoints(method_dict, n):

    method = method_dict['method']
    dist = method_dict['distribution']

    if dist._str() == 'Amalia windrose':
        # Modify the input range to start at max probability location
        # and account for zero probability regions.

        # f(x)
        #   |                   *
        #   |   ***            * *      **
        #   | **   *          *   **  **  *     ***
        #   |*      *        *      **     *  **   *
        #   |        *      *               **
        #   |         *    *
        # --+----------****-----+------------------+--
        #  lo          A  B     C                  hi    (x)
        bnd = dist.range()
        a = bnd[0]  # left boundary
        b = bnd[1]  # right boundary
        a = a[0] # get rid of the list
        b = b[0] # get rid of the list
        # Make sure the A, B, C values are the same than those in distribution
        A = 110  # Left boundary of zero probability region
        B = 140  # Right boundary of zero probability region
        C = 225  # Location of max probability
        r = b-a  # original range
        R = r - (B-A) # modified range
        dx = R/n
        # Modify with offset, manually choose the offset you want
        N = 5
        i = 0  # [-2, -1, 0, 1, 2] choose from for N=5, for general N [-int(np.floor(N/2)), ... , int(np.floor(N/2)+1]
        offset = i*dx/N
        bounds = [a+offset, R+offset]
        x = np.linspace(bounds[0], bounds[1], n+1)
        x = x[:-1]+dx/2  # Take the midpoints of the bins

        # Modify x, to start from the max probability location
        x = modifyx(x, A, B, C, r)

        if method == 'dakota':
            # Update dakota file with desired number of sample points
            # Use the x to set the abscissas, and the pdf to set the ordinates
            y = np.linspace(bounds[0], bounds[1], 51)  # play with the number here
            dy = y[1]-y[0]
            mid = y[:-1]+dy/2
            ynew = modifyx(mid, A, B, C, r)
            # print ynew
            f = dist.pdf(ynew)
            # print f*R

            # Modify y to zero to 1 range, I think makes dakota generation of polynomials easier
            y = 2*y / 330 - 1
            updateDakotaFile(method_dict['dakota_filename'], n, y, f)
            # run Dakota file to get the points locations
            x, wd = getSamplePoints(method_dict['dakota_filename'])
            # Rescale x
            print(x)
            x = 330/2. + 330/2.*x
            # Call modify x with the new x. Here also account for the offset.
            # print x
            x = modifyx(x, A, B, C, r)


        # Get the weights associated with the points locations

        if method == 'rect':
            w = getWeights(x, dx, dist)
        elif method == 'dakota':
            w = wd
        # if method == 'dakota':
        #     # Logic to get the weights from integrating the pdf between the bins
        #     w = []
        #     for i, xi in enumerate(x):
        #         if i == 0:
        #             dxleft = x[i] - C
        #             dxright = (x[i+1] - x[i])/2.
        #         elif i == (len(x)-1):
        #             dxleft = (x[i] - x[i-1])/2.
        #             dxright = C - x[i]
        #         else:
        #             dxleft = x[i] - x[i-1]
        #             dxright = x[i+1] - x[i]
        #             if dxleft < 0.0:
        #                 dxleft = dxleft + 360
        #             if dxright < 0.0:
        #                 dxright = dxright + 360
        #             dxleft = dxleft/2.
        #             dxright = dxright/2.
        #         xleft = xi-dxleft
        #         xright = xi+dxright
        #         if xright > 360.0:
        #             w.append(1 - dist._cdf(xleft) + dist._cdf(xright-360))
        #         elif xleft < 0.0:
        #             w.append(dist._cdf(xright) + (1 - dist._cdf(360+xleft)))
        #         else:
        #             w.append(dist._cdf(xright) - dist._cdf(xleft))
        #
        #     w = np.array(w).flatten()
        #     # print w  # all weights should be positive
        #     # print np.sum(w)  # this should sum to 1
        #     # print np.sum(wd)  # this should sum to 1
        #
        #     w = w#*wd  # Modify the weight with with the dakota integration weight
        #     # print np.sum(w)
        #     # w = w*R   # Modify with the range, the effect of this gets undone withing dakota (This is due to the "tricking" of the problem)
        #     # w = w*R/len(w)
        #     w = w*len(w)

        points = x
        weights = w
        return points, weights

    else:
        # Don't modify the range at all.
        bnd = dist.range()
        a = bnd[0]
        b = bnd[1]
        dx = (b-a)/n
        x = np.linspace(a+dx/2, b-dx/2, n) # Maybe modify this and then take the midpoints.

        # Get the weights associated with the points locations
        w = []
        for xi in x:
            w.append(dist._cdf(xi+dx/2.) - dist._cdf(xi-dx/2.))
        w = np.array(w).flatten()
    # return [x], w


def getLayout(method_dict):
     ### Set up the farm ###

    # Set up position of each turbine

    # Find the bounds of the amalia wind farm to 2 significant digits
    # Use this information to generate the other layouts
    # Amalia wind farm
    locations = np.genfromtxt('../WindFarms/layout_amalia.txt', delimiter=' ')
    turbineX = locations[:,0]
    turbineY = locations[:,1]

    # Find the bounds of the amalia wind farm to 2 significant digits
    round_sig = lambda x, sig=2: np.round(x, sig-int(np.floor(np.log10(x)))-1)
    xlim = round_sig(np.max(turbineX))
    ylim = round_sig(np.max(turbineY))

    layout = method_dict['layout']
    if layout == 'grid':

        # Grid farm (same number of turbines as Amalia 60)
        nRows = 10   # number of rows and columns in grid
        nCols = 6
        # spacing = 5  # turbine grid spacing in diameters, original spacing for the grid
        spacingX = xlim/(nCols)
        spacingY = ylim/(nRows)
        pointsx = np.linspace(start=0, stop=nCols*spacingX, num=nCols)
        pointsy = np.linspace(start=0, stop=nRows*spacingY, num=nRows)
        xpoints, ypoints = np.meshgrid(pointsx, pointsy)
        turbineX = np.ndarray.flatten(xpoints)
        turbineY = np.ndarray.flatten(ypoints)

    elif layout == 'random':

        # Random farm
        np.random.seed(101)
        turbineX = np.random.rand(60)*xlim
        turbineY = np.random.rand(60)*ylim

    elif layout == 'lhs':

        # Latin Hypercube farm
        np.random.seed(101)
        distx = cp.Uniform(0, xlim)
        disty = cp.Uniform(0, ylim)
        dist = cp.J(distx, disty)
        x = dist.sample(60, 'L')
        turbineX = x[0]
        turbineY = x[1]

    elif layout == 'amalia':

        # Amalia wind farm
        locations = np.genfromtxt('../WindFarms/layout_amalia.txt', delimiter=' ')
        turbineX = locations[:,0]
        turbineY = locations[:,1]

    elif layout == 'optimized':

        # Amalia optimized Jared
        locations = np.genfromtxt('../WindFarms/AmaliaOptimizedXY.txt', delimiter=' ')
        turbineX = locations[:,0]
        turbineY = locations[:,1]

    else:
        raise ValueError('unknown layout option "%s", \nvalid options ["amalia", "optimized", "random", "lhs", "grid"]' %layout)

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

    return turbineX, turbineY


if __name__ == "__main__":

    # config.floris_single_component = True  I don't think this doing anything

    #########################################################################

    method_dict = {}
    method_dict['method']           = 'dakota'
    method_dict['uncertain_var']    = 'direction'
    method_dict['layout']           = 'optimized'

    if method_dict['uncertain_var'] == 'speed':
        dist = distributions.getWeibull()
        method_dict['distribution'] = dist
    elif method_dict['uncertain_var'] == 'direction':
        dist = distributions.getWindRose()
        method_dict['distribution'] = dist
    else:
        raise ValueError('unknown uncertain_var option "%s", valid options "speed" or "direction".' %method_dict['uncertain_var'])

    method_dict['dakota_filename'] = 'dakotageneral.in'

    # n = 10  # number of processors (and number of wind directions to run)
    n = 10

    points, weights = getPoints(method_dict, n)

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

    # For not changing the names right now.
    windSpeeds = windspeeds
    windDirections = winddirections

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

    # Here I actually pick the layouts I want. These are larger
    # turbineX, turbineY = getLayout(method_dict)


    locations = np.column_stack((turbineX, turbineY))
    # generate boundary constraint
    boundaryVertices, boundaryNormals = calculate_boundary(locations)
    nVertices = boundaryVertices.shape[0]
    print('boundary vertices', boundaryVertices)

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
    prob = Problem(root=OptAEP(nTurbines=nTurbs, nDirections=windDirections.size, minSpacing=minSpacing, use_rotor_components=False, differentiable=True, nVertices=nVertices, method_dict=method_dict))

    # set up optimizer
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.add_objective('obj', scaler=1E-8)  # the amalia has the scaler at 1e-5

    # set optimizer options
    prob.driver.opt_settings['Verify level'] = 3
    prob.driver.opt_settings['Print file'] = 'SNOPT_print_exampleOptAEP.out'
    prob.driver.opt_settings['Summary file'] = 'SNOPT_summary_exampleOptAEP.out'
    prob.driver.opt_settings['Major iterations limit'] = 1000
    prob.driver.opt_settings['Major optimality tolerance'] = 2E-6


    # select design variables
    prob.driver.add_desvar('turbineX', scaler=1.0)
    prob.driver.add_desvar('turbineY', scaler=1.0)
    # for direction_id in range(0, windDirections.size):
    #     prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1.0)
    # prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs)*min(turbineX), upper=np.ones(nTurbs)*max(turbineX), scaler=1E-2)
    # prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY), scaler=1E-2)
    # for direction_id in range(0, windDirections.size):
    #     prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1E-1)
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
    for direction_id in range(0, windDirections.size):
        prob['yaw%i' % direction_id] = yaw

    # assign values to constant inputs (not design variables)
    prob['windSpeeds'] = windSpeeds
    prob['windDirections'] = windDirections
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
    print('wind farm power in each direction (kW): %s' % prob['power'])
    print('AEP (kWh): %s' % prob['mean'])

    xbounds = [min(turbineX), min(turbineX), max(turbineX), max(turbineX), min(turbineX)]
    ybounds = [min(turbineY), max(turbineY), max(turbineY), min(turbineY), min(turbineX)]

    np.savetxt('AmaliaOptimizedXY.txt', np.c_[prob['turbineX'], prob['turbineY']], header="turbineX, turbineY")

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


