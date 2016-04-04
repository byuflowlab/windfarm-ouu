
import subprocess
import sys
import numpy as np
from dakotaInterface import RedirectOutput


def getSamplePoints(dakotaFile):
    """Call Dakota to get the sample points.

    Args:
        dakotaFile (string): The dakota input file

    Returns:
        x (np.array): A vector of sample points

    """

    print 'Calling Dakota...'
    # Pipe the output
    log = 'logDakota.out'
    err = log  # will append the error to the output
    with RedirectOutput(log, err):
        dakotaInput = dakotaFile
        # dakotaInput = '--version'
        subprocess.check_call(['dakota', dakotaInput], stdout=sys.stdout,
                              stderr=sys.stderr)

    print 'finished calling Dakota.'

    # read the points from the dakota quadrature tabular file (Only prints when running verbose)
    dakotaTabular = 'dakota_tabular.dat'
    f = open(dakotaTabular, 'r')
    f.readline()
    x = []
    for line in f:
        x.append(float(line.split()[2]))

    x = np.array(x)

    return x


def getSamplePoints2(dakotaFile):
    """Call Dakota to get the sample points.

    Args:
        dakotaFile (string): The dakota input file

    Returns:
        x (np.array): A vector of sample points
        w (np.array): The weight associated with the sample points

    """

    print 'Calling Dakota...'
    # Pipe the output
    log = 'logDakota.out'
    err = log  # will append the error to the output
    with RedirectOutput(log, err):
        dakotaInput = dakotaFile
        # dakotaInput = '--version'
        subprocess.check_call(['dakota', dakotaInput], stdout=sys.stdout,
                              stderr=sys.stderr)

    print 'finished calling Dakota.'

    # read the points from the dakota quadrature tabular file (Only prints when running verbose)
    dakotaTabular = 'dakota_quadrature_tabular.dat'
    f = open(dakotaTabular, 'r')
    f.readline()
    x = []
    w = []
    for line in f:
        w.append(float(line.split()[1]))
        x.append(float(line.split()[2]))

    x = np.array(x)
    w = np.array(w)

    # Scale the sample points to their true range from the [-1,1] range.
    # This is for uniform_uncertain variables
    f = open(dakotaFile, 'r')
    for line in f:
        if line.lstrip().startswith('#') and 'uniform_uncertain' in line:
            a = -1.0
            b = 1.0
            break
        if 'lower_bounds' in line:
            a = float(line.split()[2])  # will need to update for multiple variables
        if 'upper_bounds' in line:
            b = float(line.split()[2])
    f.close()

    x = (a+b)/2 + (b-a)/2*x

    return x, w

if __name__ == '__main__':
    dakotaFileName = 'dakotaAEPdirection.in'
    points, weights = getSamplePoints(dakotaFileName)
    print 'points = ', points
    print 'weights = ', weights
