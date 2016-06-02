import numpy as np
# import matplotlib.pyplot as plt
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

        # Modify with offset, manually choose the offset you want
        N = 5
        i = 0  # [-2, -1, 0, 1, 2] choose from for N=5, for general N [-int(np.floor(N/2)), ... , int(np.floor(N/2)+1]

        if method == 'rect':
            # the offset fits N points in the given dx interval
            dx = R/n
            offset = i*dx/N
            bounds = [a+offset, R+offset]
            x = np.linspace(bounds[0], bounds[1], n+1)
            x = x[:-1]+dx/2  # Take the midpoints of the bins
            print x
            # Modify x, to start from the max probability location
            x = modifyx(x, A, B, C, r)
            # Get the weights associated with the points locations
            w = getWeights(x, dx, dist)

        if method == 'dakota':
            # the offset modifies the starting point for 5 locations within the whole interval
            # Update dakota file with desired number of sample points
            # Use the y to set the abscissas, and the pdf to set the ordinates
            y = np.linspace(a, R, 51)  # play with the number here
            dy = y[1]-y[0]
            mid = y[:-1]+dy/2
            # Modify the starting point C with offset
            offset = i*r/N
            C = (C + offset) % r
            ynew = modifyx(mid, A, B, C, r)
            f = dist.pdf(ynew)

            # Modify y to -1 to 1 range, I think makes dakota generation of polynomials easier
            y = 2*y / 330 - 1
            updateDakotaFile(method_dict['dakota_filename'], n, y, f)
            # run Dakota file to get the points locations
            x, wd = getSamplePoints(method_dict['dakota_filename'])
            # Rescale x
            x = 330/2. + 330/2.*x
            # Call modify x with the new x.
            x = modifyx(x, A, B, C, r)
            # Get the weights associated with the points locations
            w = wd

        points = x
        weights = w

    else:  # This is mostly for speed case
        # Don't modify the range at all.
        bnd = dist.range()
        a = bnd[0]
        b = bnd[1]
        a = a[0]  # get rid of the list
        b = b[0]  # get rid of the list

        if method == 'rect':
            # the offset fits N points in the given dx interval
            # Modify with offset, manually choose the offset you want
            N = 5
            i = 0  # [-2, -1, 0, 1, 2] choose from for N=5, for general N [-int(np.floor(N/2)), ... , int(np.floor(N/2)+1]
            dx = (b-a)/n
            offset = i*dx/N
            bounds = [a+offset, b+offset]
            x = np.linspace(bounds[0], bounds[1], n+1)
            x = x[:-1]+dx/2  # Take the midpoints of the bins
            print x
            # Get the weights associated with the points locations
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


        if method == 'dakota':
            # The offset doesn't really make sense for this case
            # Update dakota file with desired number of sample points
            # Use the y to set the abscissas, and the pdf to set the ordinates
            y = np.linspace(a, b, 51)  # play with the number here
            dy = y[1]-y[0]
            ymid = y[:-1]+dy/2
            f = dist.pdf(ymid)
            # Modify y to -1 to 1 range, I think makes dakota generation of polynomials easier
            y = 2*y / 30 - 1

            updateDakotaFile(method_dict['dakota_filename'], n, y, f)
            # run Dakota file to get the points locations
            x, wd = getSamplePoints(method_dict['dakota_filename'])
            # Rescale x
            x = 30/2. + 30/2.*x

            # Get the weights associated with the points locations
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

    if layout == 'grid':

        # Grid wind farm
        locations = np.genfromtxt('../WindFarms/layout_grid.txt', delimiter=' ')
        turbineX = locations[:, 0]
        turbineY = locations[:, 1]

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
        locations = np.genfromtxt('../WindFarms/layout_random.txt', delimiter=' ')
        turbineX = locations[:, 0]
        turbineY = locations[:, 1]

    elif layout == 'amalia':

        # Amalia wind farm
        locations = np.genfromtxt('../WindFarms/layout_amalia.txt', delimiter=' ')
        turbineX = locations[:, 0]
        turbineY = locations[:, 1]

    elif layout == 'optimized':

        # Amalia optimized
        # locations = np.genfromtxt('../WindFarms/AmaliaOptimizedXY.txt', delimiter=' ') # Amalia optimized Jared
        locations = np.genfromtxt('../WindFarms/layout_optimized.txt', delimiter=' ')
        turbineX = locations[:,0]
        turbineY = locations[:,1]

    else:
        raise ValueError('unknown layout option "%s", \nvalid options ["amalia", "optimized", "random", "test", "grid"]' %layout)

    # plt.figure()
    # plt.scatter(turbineX, turbineY)
    # plt.show()

    return turbineX, turbineY
