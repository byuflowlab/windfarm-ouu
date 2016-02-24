
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

    # read the points from the dakota tabular file
    # dakotaTabular = 'dakota_tabular.dat'
    dakotaTabular = 'dakota_quadrature_tabular.dat'
    f = open(dakotaTabular, 'r')
    f.readline()
    x = []
    w = []
    for line in f:
        w.append(float(line.split()[1]))
        x.append(float(line.split()[2]))
    return np.array(x), np.array(w)

if __name__ == '__main__':
    # dakotaFileName = 'dakotaSamplePoints.in'
    dakotaFileName = 'dakotaAEP.in'
    points, weights = getSamplePoints(dakotaFileName)
    print 'points = ', points
    print 'weights = ', weights
