
import subprocess
import sys
import numpy as np
from dakotaInterface import RedirectOutput

def getDakotaStatistics(dakotaFile):

    dakotaInput = dakotaFile

    print 'Calling Dakota...'
    # Pipe the output
    log = 'logDakota.out'
    err = log  # will append the error to the output
    with RedirectOutput(log, err):
        # dakotaInput = '--version'
        subprocess.check_call(['dakota', dakotaInput], stdout=sys.stdout,
                              stderr=sys.stderr)

    print 'finished calling Dakota.'

    # Postprocess the results
    mean, std, coeff = postprocess()
    return mean, std, coeff


def postprocess():
    """Read the Mean and the coefficients from the Dakota output."""

    filename = 'logDakota.out'

    with open(filename, 'r') as f:
        # Find the coefficients
        # while True:
        #     line = f.readline()
        #     if 'coefficient' in line:
        #         break
        #     if not line: break
        #
        # # Read the coefficients
        # f.readline()
        coeff = []
        # while True:
        #     try:
        #         coeff.append(float(f.readline().split()[0]))
        #     except ValueError:
        #         break

        # Find the function
        while True:
            line = f.readline()
            if 'Mean' in line:
                # function value
                # f.readline()  # for the least squares case
                line = f.readline()
                mean = float(line.split()[1])
                std = float(line.split()[2])
                break
            if not line: break

    return np.array(mean), np.array(std), np.array(coeff)


if __name__ == '__main__':

    # dakotaFileName = 'dakotaAEP.in'
    dakotaFileName = sys.argv[1]

    mean, std, coeff = getDakotaStatistics(dakotaFileName)
    # print 'mean', mean
    # print 'chaos coefficients', coeff

    # Write out the calculated AEP to be read by the DakotaAEP Component
    np.savetxt('mean.txt', [mean], header='mean power')  # put in [] It doesn't like to write a scalar
    np.savetxt('std.txt', [std], header='std power')