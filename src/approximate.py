import numpy as np
import matplotlib.pyplot as plt
import json
import distributions
import windfarm_setup


def generate_approx_file(uncertain_variable, approxfile='approximate_at.dat'):
    """Generate the file at which to evaluate the PC approximation."""

    n = 100  # Number of approximation points per dimension

    if uncertain_variable == 'direction_and_speed':
        N = 2  # we have two uncertain variables
    else:
        N = 1  # we have one uncertain variable

    f = open(approxfile, 'w')
    if N == 1:
        f.write('%eval_id interface\t x1 \n')
    elif N == 2:
        f.write('%eval_id interface\t x1\t x2 \n')
    x = np.linspace(-1, 1, n+1)
    # Take the midpoints
    dx = x[1]-x[0]
    x = x[:-1] + dx/2

    if N == 1:
        for i in range(n):
            line = str(i+1) + '\t' + 'APPROX_INTERFACE' + '\t' + str(x[i]) + '\n'
            f.write(line)
    if N == 2:
        for i in range(n):
            for j in range(n):
                line = str((i*n)+j+1) + '\t' + 'APPROX_INTERFACE' + '\t' + str(x[i]) + '\t' + str(x[j]) + '\n'
                f.write(line)

    f.close()

    # print 'wrote ' + approxfile


def read_the_approx_file(approxfile='approximated.dat'):
    f = open(approxfile, 'r')
    line = f.readline()
    n = len(line.split())
    x = [[] for i in range(2, n-1)]  # create list to hold the variables
    p = []
    for line in f:
        splitline = line.split()
        for i in range(len(x)):
            x[i].append(float(splitline[2+i]))
        p.append(float(splitline[-1]))

    f.close()
    return x, p


def get_approximation(method_dict):

    # read the points from the dakota approximation file
    # Make sure this file was set in the dakota input.
    dakotaApprox = 'approximated.dat'
    x, p = read_the_approx_file(dakotaApprox)

    # Modify the points according to the distribution
    uncertain_var = method_dict['uncertain_var']
    if uncertain_var == 'speed':
        assert len(x) == 1, 'Should only be returning the speeds'
        x = np.array(x[0])
        p = np.array(p)

        dist = method_dict['distribution']
        bnd = dist.range()
        a = bnd[0]  # lower boundary
        b = bnd[1]  # upper boundary
        a = a[0]  # get rid of the list
        b = b[0]  # get rid of the list
        # Rescale x
        x = (b-a)/2. + (b-a)/2.*x + a

        windspeed_approx = x
        winddirection_approx = np.array([method_dict['winddirection_ref']])
        power_approx = p


    elif uncertain_var == 'direction':

        assert len(x) == 1, 'Should only be returning the directions'
        x = np.array(x[0])
        p = np.array(p)

        dist = method_dict['distribution']

        # The original amalia windrose make this function later.
        if dist._str() == 'Amalia windrose':

            bnd = dist.range()
            a = bnd[0]  # left boundary
            b = bnd[1]  # right boundary
            a = a[0]  # get rid of the list
            b = b[0]  # get rid of the list
            # Make sure the A, B, C values are the same than those in distribution
            A, B = dist.get_zero_probability_region()
            # A = 110  # Left boundary of zero probability region
            # B = 140  # Right boundary of zero probability region

            C = 225  # Location of max probability or desired starting location.
            r = b-a  # original range
            R = r - (B-A) # modified range

            # Modify with offset, manually choose the offset you want
            N = method_dict['Noffset']  # N = 10
            i = method_dict['offset']  # i = [0, 1, 2, N-1]

            # Rescale x
            x = R*x/2. + R/2. + a
            # Modify the starting point C with offset
            offset = i*r/N  # the offset modifies the starting point for N locations within the whole interval
            C = (C + offset) % r
            x = windfarm_setup.modifyx(x, A, B, C, r)

        if dist._str() == 'Amalia windrose raw':

            bnd = dist.range()
            a = bnd[0]  # left boundary
            b = bnd[1]  # right boundary
            a = a[0]  # get rid of the list
            b = b[0]  # get rid of the list

            C = 225  # Location of max probability or desired starting location.
            R = b-a  # range 360

            # Modify with offset, manually choose the offset you want
            N = method_dict['Noffset']  # N = 10
            i = method_dict['offset']  # i = [0, 1, 2, N-1]

            # Rescale x
            x = R*x/2. + R/2. + a
            # Modify the starting point C with offset
            offset = i*R/N  # the offset modifies the starting point for N locations within the whole interval
            C = (C + offset) % R
            x = (x+C) % R

        # Rearrange for plotting
        order = x.argsort()
        x = x[order]
        p = p[order]

        windspeed_approx = np.array([method_dict['windspeed_ref']])
        winddirection_approx = x
        power_approx = p

    elif uncertain_var == 'direction_and_speed':
        assert len(x) == 2, 'Should be returning the directions and speeds'
        x_d = np.array(x[0])
        x_s = np.array(x[1])
        p = np.array(p)

        dist = method_dict['distribution']

        bnd = dist.range()
        a = bnd[0]  # left boundary
        b = bnd[1]  # right boundary
        a_d = a[0] # the left boundary for the direction
        b_d = b[0] # the right boundary for the direction
        a_s = a[1] # the left boundary for the direction
        b_s = b[1] # the right boundary for the direction

        ###### Do direction work
        dist_dir = dist[0]

        # The original amalia windrose make this function later. I still need to make this for the raw
        if dist_dir._str() == 'Amalia windrose':

            # Make sure the A, B, C values are the same than those in distribution
            A, B = dist_dir.get_zero_probability_region()
            # A = 110  # Left boundary of zero probability region
            # B = 140  # Right boundary of zero probability region

            C = 225  # Location of max probability
            r = b_d-a_d  # original range
            R = r - (B-A) # modified range

            # Modify with offset, manually choose the offset you want
            N = method_dict['Noffset']  # N = 10
            i = method_dict['offset']  # i = [0, 1, 2, N-1]

            # Rescale x
            x_d = R*x_d/2. + R/2. + a_d
            # Modify the starting point C with offset
            offset = i*r/N  # the offset modifies the starting point for N locations within the whole interval
            C = (C + offset) % r
            x_d = windfarm_setup.modifyx(x_d, A, B, C, r)

        if dist_dir._str() == 'Amalia windrose raw':

            C = 225  # Location of max probability or desired starting location.
            R = b_d-a_d  # range 360

            # Modify with offset, manually choose the offset you want
            N = method_dict['Noffset']  # N = 10
            i = method_dict['offset']  # i = [0, 1, 2, N-1]

            # Rescale x
            x_d = R*x_d/2. + R/2. + a_d
            # Modify the starting point C with offset
            offset = i*R/N  # the offset modifies the starting point for N locations within the whole interval
            C = (C + offset) % R
            x_d = (x_d+C) % R

        ####### Do the speed work
        # Rescale x
        x_s = (b_s-a_s)/2. + (b_s-a_s)/2.*x_s + a_s

        # Rearrange for plotting
        order = np.lexsort((x_s, x_d))  # Sort by x_d, then by x_s
        # print 'order = ', order
        x_s = x_s[order]
        x_d = x_d[order]
        p = p[order]

        windspeed_approx = x_s
        winddirection_approx = x_d
        power_approx = p

    else:
        raise ValueError('unknown uncertain_var option "%s", valid options "speed" or "direction".' %uncertain_var)

    return winddirection_approx, windspeed_approx, power_approx


if __name__ == "__main__":

    # method_dict = {'uncertain_var': 'speed', 'distribution': distributions.getWeibull()}
    method_dict = {'uncertain_var': 'direction', 'distribution': distributions.getWindRose(),
               'Noffset': 10, 'offset': 0, 'windspeed_ref': 8,
                   'winddirection_ref': 225}


    # Get the approximation points
    d, s, p = get_approximation(method_dict)

    # Get the reference for comparison
    f = open('figure1.json', 'r')
    r = json.load(f)
    f.close()

    if method_dict['uncertain_var'] == 'speed':
        pref = np.array(r['speed_optimized']['power'])
        xref = np.array(r['speed_optimized']['speed'])
        x = s
    elif method_dict['uncertain_var'] == 'direction':
        pref = np.array(r['dir_optimized']['power'])
        xref = np.array(r['dir_optimized']['direction'])
        x = d

    fig, ax = plt.subplots()
    ax.plot(xref, pref/1e3, label='actual')
    ax.plot(x, p/1e3, label='pc approx')
    ax.legend()

    plt.show()

