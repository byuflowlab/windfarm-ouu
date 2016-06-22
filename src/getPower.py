#!/usr/bin/env python

# Read DAKOTA parameters file standard format
# Call your application for analysis
# Return the results file to Dakota.

# DAKOTA will execute this script as
#   getPoints.py params.in results.out
#   so sys.argv[1] will be the parameters file and
#   sys.argv[2] will be the results file to return to DAKOTA

# necessary python modules
import sys
import dakotaInterface
import numpy as np


def main():

    # ----------------------------
    # Parse DAKOTA parameters file
    # ----------------------------
    paramsfile = sys.argv[1]
    paramsdict = dakotaInterface.parseDakotaParametersFile(paramsfile)

    # -------- Modify here for your problem -------- #

    # ------------------------
    # Set up your application
    # ------------------------

    nVarUncertain = 1
    dakotaInterface.checknVar(nVarUncertain, paramsdict)

    active_set_vector = [int(paramsdict['ASV_1:power'])]

    # -----------------------------
    # Execute your application
    # -----------------------------

    try:
        power = np.atleast_1d(np.loadtxt('powerInput.txt'))
        index = int(paramsdict['eval_id']) - 1
        power_i = power[index]

    except IOError:  # This is for the case when we are only getting the sample points.
        print '\n\nWARNING: missing powerInput.txt\n\n'
        power_i = -1.0  # np.nan

    resultsdict = {'fns': [power_i], 'fnGrads': []}

    # ----------------------------
    # Return the results to DAKOTA
    # ----------------------------

    resultsfile = sys.argv[2]
    dakotaInterface.writeDakotaResultsFile(
        resultsfile, resultsdict, paramsdict, active_set_vector)


if __name__ == '__main__':
    main()
