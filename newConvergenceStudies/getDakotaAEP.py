
import subprocess
import sys
import numpy as np
from dakotaInterface import RedirectOutput

def getDakotaAEP(dakotaFile):

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
    AEP, coeff = postprocess()
    return AEP, coeff


def postprocess():
    """Read the Mean and the coefficients from the Dakota output."""

    filename = 'logDakota.out'

    with open(filename, 'r') as f:
        # Find the coefficients
        while True:
            line = f.readline()
            if 'coefficient' in line:
                break
            if not line: break

        # Read the coefficients
        f.readline()
        coeff = []
        while True:
            try:
                coeff.append(float(f.readline().split()[0]))
            except ValueError:
                break

        # Find the function
        while True:
            line = f.readline()
            if 'Mean' in line:
                # function value
                f.readline()
                line = f.readline()
                AEP = float(line.split()[1])
                break
            if not line: break

    return np.array(AEP), np.array(coeff)


if __name__ == '__main__':

    # dakotaFileName = 'dakotaAEP.in'
    dakotaFileName = sys.argv[1]

    AEP, coeff = getDakotaAEP(dakotaFileName)
    # print 'AEP', AEP
    # print 'chaos coefficients', coeff

    # Write out the calculated AEP to be read by the DakotaAEP Component
    np.savetxt('AEP.txt', [AEP], header='AEP')  # put in [] It doesn't like to write a scalar