
import subprocess
import sys
import numpy as np
from dakotaInterface import RedirectOutput, updateDakotaFile


def getDakotaAEP(dakotaFile):

    designVars = getTurbineLocations()

    dakotaInput = dakotaFile
    updateDakotaFile(dakotaInput, designVars)

    print 'Calling Dakota...'
    # Pipe the output
    log = 'logDakota.out'
    err = log  # will append the error to the output
    with RedirectOutput(log, err):
        # dakotaInput = '--version'
        subprocess.check_call(['dakota', dakotaInput], stdout=sys.stdout,
                              stderr=sys.stderr)

    print 'finished calling Dakota.'

    # postprocess (This should also go in the Dakota component)
    AEP, AEPgrad, coeff = postprocess()
    return AEP, AEPgrad, coeff


def getTurbineLocations():

    # This will be replaced in the preprocess step of the
    # Dakota component

    # define turbine size
    rotor_diameter = 1  # (m)

    # Scaling grid case
    nRows = 3    # number of rows and columns in grid
    spacing = 5     # turbine grid spacing in diameters

    # Set up position arrays
    points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineX = np.ndarray.flatten(xpoints)
    turbineY = np.ndarray.flatten(ypoints)

    return np.concatenate((turbineX, turbineY))


def postprocess():
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

        # Find the function and gradient values
        AEPgrad = []
        while True:
            line = f.readline()
            if 'mean_r1' in line:
                # function value
                AEP = float(line.split()[0])
                # get the gradient
                while True:
                    line = f.readline()
                    split_line = line.split()
                    if len(split_line) > 0:
                        if split_line[0] == '[':
                            for entry in split_line[1:]:
                                try:
                                    AEPgrad.append(float(entry))
                                except ValueError:
                                    break
                        else:
                            for entry in split_line:
                                try:
                                    AEPgrad.append(float(entry))
                                except ValueError:
                                    break
                    else:
                        break
            if not line: break

    return np.array(AEP), np.array(AEPgrad), np.array(coeff)

if __name__ == '__main__':
    dakotaFileName = 'dakotaAEP.in'
    AEP, AEPgrad, coeff = getDakotaAEP(dakotaFileName)
    print 'AEP', AEP
    print 'AEPgrad', AEPgrad
    print 'chaos coefficients', coeff