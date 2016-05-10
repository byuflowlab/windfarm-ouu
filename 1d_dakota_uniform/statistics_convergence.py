
import numpy as np
import matplotlib.pyplot as plt
import chaospy as cp
import json
from openmdao.api import Problem
from AEPGroups import AEPGroup
from getSamplePoints import getSamplePoints
from dakotaInterface import updateDakotaFile, updateDakotaFile2
import distributions
import quadrature_rules


def run():
    """
    method_dict = {}
    keys of method_dict:
        'method' = 'dakota', 'rect' or 'chaospy'  # 'chaospy needs updating
        'uncertain_var' = 'speed' or 'direction'
        'layout' = 'amalia', 'optimized', 'grid', 'random', 'lhs'
        'dakota_filename' = 'dakotaInput.in', applicable for dakota method
        'distribution' = a distribution applicable for rect and chaospy methods, it gets set in getPoints()
    Returns:
        Writes a json file 'record.json' with the run information.
    """

    method_dict = {}
    method_dict['method']           = 'rect'
    method_dict['uncertain_var']    = 'direction'
    method_dict['layout']             = 'lhs'

    if method_dict['uncertain_var'] == 'speed':
        dist = distributions.getWeibull()
        method_dict['distribution'] = dist
    elif method_dict['uncertain_var'] == 'direction':
        dist = distributions.getWindRose()
        method_dict['distribution'] = dist
    else:
        raise ValueError('unknown uncertain_var option "%s", valid options "speed" or "direction".' %method_dict['uncertain_var'])

    method_dict['dakota_filename'] = 'dakotageneral.in'

    mean = []
    std = []
    samples = []

    for n in range(5,6,1):

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


        print 'Locations at which power is evaluated'
        print '\twindspeed \t winddirection'
        for i in range(n):
            print i+1, '\t', '%.2f' % windspeeds[i], '\t', '%.2f' % winddirections[i]

        # Set up problem, define the turbine locations and all that stuff
        prob = problem_set_up(windspeeds, winddirections, weights, method_dict)

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
        x = np.linspace(a, R, n+1)
        dx = x[1]-x[0]  # I could also get dx from dx = R/n
        # Modify with offset, manually choose the offset you want
        N = 5
        i = 0  # [-2, -1, 0, 1, 2] choose from for N=5, for general N [-int(np.floor(N/2)), ... , int(np.floor(N/2)+1]
        offset = i*dx/N
        bounds = [a+offset, R+offset]
        if method == 'rect':
            x = np.linspace(bounds[0], bounds[1], n+1)
            x = x[:-1]+dx/2  # Take the midpoints of the bins
        if method == 'dakota':
            # Update dakota file with desired number of sample points
            updateDakotaFile(method_dict['dakota_filename'], n, bounds)
            # run Dakota file to get the points locations
            x, wd = getSamplePoints(method_dict['dakota_filename'])

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
        x = np.array(y)

        # Get the weights associated with the points locations

        if method == 'rect':
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
                print xi+dx/2., xi-dx/2.
            w = np.array(w).flatten()
            print w  # all weights should be positive
            print np.sum(w)   # this should sum to 1

        if method == 'dakota':
            # Logic to get the weights from integrating the pdf between the bins
            w = []
            for i, xi in enumerate(x):
                if i == 0:
                    dxleft = x[i] - C
                    dxright = (x[i+1] - x[i])/2.
                elif i == (len(x)-1):
                    dxleft = (x[i] - x[i-1])/2.
                    dxright = C - x[i]
                else:
                    dxleft = x[i] - x[i-1]
                    dxright = x[i+1] - x[i]
                    if dxleft < 0.0:
                        dxleft = dxleft + 360
                    if dxright < 0.0:
                        dxright = dxright + 360
                    dxleft = dxleft/2.
                    dxright = dxright/2.
                xleft = xi-dxleft
                xright = xi+dxright
                if xright > 360.0:
                    w.append(1 - dist._cdf(xleft) + dist._cdf(xright-360))
                elif xleft < 0.0:
                    w.append(dist._cdf(xright) + (1 - dist._cdf(360+xleft)))
                else:
                    w.append(dist._cdf(xright) - dist._cdf(xleft))

            w = np.array(w).flatten()
            print w  # all weights should be positive
            print np.sum(w)  # this should sum to 1
            print np.sum(wd)  # this should sum to 1

            w = w#*wd  # Modify the weight with with the dakota integration weight
            # print np.sum(w)
            # w = w*R   # Modify with the range, the effect of this gets undone withing dakota (This is due to the "tricking" of the problem)
            # w = w*R/len(w)
            w = w*len(w)

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


def problem_set_up(windspeeds, winddirections, weights, method_dict=None):
    """Set up wind farm problem.

    Args:
        windspeeds (np.array): wind speeds vector
        winddirections (np.array): wind directions vector
        weights (np.array): integration weights associated with the windspeeds and winddirections
        method_dict (dict): UQ method and parameters for the UQ method

    Returns:
        prob (openMDAO problem class): The set up wind farm.

    """




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


    plt.figure()
    plt.scatter(turbineX, turbineY)
    plt.show()

    # turbine size and operating conditions

    rotor_diameter = 126.4  # (m)
    air_density = 1.1716    # kg/m^3

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
                            weights=weights, method_dict=method_dict))

    # initialize problem
    prob.setup(check=False)

    # assign initial values to variables
    prob['windSpeeds'] = windspeeds
    prob['windDirections'] = winddirections
    prob['weights'] = weights
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


