
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
        x (list): contains vectors (np.array) of the sample points

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
    x1 = []
    x2 = []
    for line in f:
        x1.append(float(line.split()[2]))
        x2.append(float(line.split()[3]))
    print x1
    print x2
    x = [np.array(x1), np.array(x2)]
    print 'aloha', x
    return x

if __name__ == '__main__':
    dakotaFileName = 'dakotaAEPdirection.in'
    points, weights = getSamplePoints(dakotaFileName)
    print 'points = ', points
    print 'weights = ', weights
