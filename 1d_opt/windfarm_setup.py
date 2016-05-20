import numpy as np
import chaospy as cp

from getSamplePoints import getSamplePoints
from dakotaInterface import updateDakotaFile



def getPoints(method_dict, n):

    method = method_dict['method']
    dist = method_dict['distribution']

    if dist._str() == 'Amalia windrose':  # For direction case
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

            # Modify y to -1 to 1 range, I think makes dakota generation of polynomials easier
            y = 2*y / 330 - 1
            updateDakotaFile(method_dict['dakota_filename'], n, y, f)
            # run Dakota file to get the points locations
            x, wd = getSamplePoints(method_dict['dakota_filename'])
            # Rescale x
            # print x
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

    else:  # This is mostly for speed case
        # Don't modify the range at all.
        bnd = dist.range()
        a = bnd[0]
        b = bnd[1]
        a = a[0]  # get rid of the list
        b = b[0]  # get rid of the list
        dx = (b-a)/n
        # x = np.linspace(a+dx/2, b-dx/2, n) # Maybe modify this and then take the midpoints.
        # Modify with offset, manually choose the offset you want
        N = 5
        i = 0  # [-2, -1, 0, 1, 2] choose from for N=5, for general N [-int(np.floor(N/2)), ... , int(np.floor(N/2)+1]
        offset = i*dx/N
        bounds = [a+offset, b+offset]
        x = np.linspace(bounds[0], bounds[1], n+1)
        x = x[:-1]+dx/2  # Take the midpoints of the bins

        if method == 'dakota':
            # Update dakota file with desired number of sample points
            # Use the x to set the abscissas, and the pdf to set the ordinates
            y = np.linspace(bounds[0], bounds[1], 51)  # play with the number here
            dy = y[1]-y[0]
            ymid = y[:-1]+dy/2
            f = dist.pdf(ymid)

            # Modify y to -1 to 1 range, I think makes dakota generation of polynomials easier
            y = 2*y / 30 - 1


            ####### Revise this to make sure it works with Dakota


            updateDakotaFile(method_dict['dakota_filename'], n, y, f)
            # run Dakota file to get the points locations
            x, wd = getSamplePoints(method_dict['dakota_filename'])
            # Rescale x
            x = 30/2. + 30/2.*x

        # Get the weights associated with the points locations

        if method == 'rect':
            w = []
            for xi in x:
                xleft = xi-dx/2.
                xright = xi+dx/2.
                if xleft < a:
                    # print 'I am in xleft'
                    xleft = a
                if xright > b:
                    # print 'I am in xright'
                    xright = b
                w.append(dist._cdf(xright) - dist._cdf(xleft))
            w = np.array(w).flatten()
            # print np.sum(w)
            # print dist._cdf(b)  # this value should weight dakota weights. b=30
        elif method == 'dakota':
            w = wd * dist._cdf(b)  # The dakota weights assume all of the pdf is between 0-30 so we weigh it by the actual amount. This will correct the derivatives, need to also correct the mean and std values. These corrections are done in statisticsComponents.


        points = x
        weights = w
        # print weights
        # print np.sum(weights)

    return points, weights


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

def getLayout(layout='grid'):
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

    elif layout == 'test':
        # Small Farm
        nRows = 2  # 10   # number of rows and columns in grid
        spacing = 5  # turbine grid spacing in diameters
        rotor_diameter = 126.4  # (m)

        # Grid farm
        points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
        xpoints, ypoints = np.meshgrid(points, points)
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