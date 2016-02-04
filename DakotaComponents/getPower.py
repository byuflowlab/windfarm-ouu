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

    nVarDesign = 18
    nVarUncertain = 1
    nVar = nVarDesign + nVarUncertain
    dakotaInterface.checknVar(nVar, paramsdict)

    wind_direction = [float(paramsdict['x'])]
    design_vars = []
    for i in range(1, nVarDesign+1):
        var = 'x' + str(i)
        design_vars.append(float(paramsdict[var]))
    active_set_vector = [int(paramsdict['ASV_1:power'])]

    # -----------------------------
    # Execute your application
    # -----------------------------
    # Need to read in the function and gradient values

    resultsdict = {'fns': wind_direction, 'fnGrads': [design_vars]}

    # ----------------------------
    # Return the results to DAKOTA
    # ----------------------------

    resultsfile = sys.argv[2]
    dakotaInterface.writeDakotaResultsFile(
        resultsfile, resultsdict, paramsdict, active_set_vector)


if __name__ == '__main__':
    main()
