
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

    # read the points from the dakota tabular file
    dakotaTabular = 'dakota_tabular.dat'
    f = open(dakotaTabular, 'r')
    f.readline()
    x = []
    for line in f:
        x.append(float(line.split()[2]))

    return np.array(x)

if __name__ == '__main__':
    dakotaFileName = 'dakotaSamplePoints.in'
    points = getSamplePoints(dakotaFileName)
    print points
