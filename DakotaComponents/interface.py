#!/usr/bin/env python

# Read DAKOTA parameters file standard format and call SU2
# python module for analysis and return the results file to Dakota.

# DAKOTA will execute this script as
#   SU2_dakota_interface.py params.in results.out
#   so sys.argv[1] will be the parameters file and
#   sys.argv[2] will be the results file to return to DAKOTA

# necessary python modules
import sys
import numpy as np
import re
import subprocess

def main():

    # ----------------------------
    # Parse DAKOTA parameters file
    # ----------------------------
    paramsfile = sys.argv[1]
    paramsdict = parse_dakota_parameters_file(paramsfile)

    # ------------------------
    # Set up application (SU2)
    # ------------------------

    ########## Modify here for your problem ##########

    nvar = 1  # Specify the number of uncertain variables
    check_nvar(nvar,paramsdict)

    wind_direction = [float(paramsdict['x'])]
    active_set_vector = [ int(paramsdict['ASV_1:dummy'])]

    # -----------------------------
    # Execute the application (SU2)
    # -----------------------------
    # This is dummy function just returns the wind direction

    resultsdict = {}
    resultsdict['fns'] = wind_direction

    #subprocess.call(["python", "exampleCallDakota.py"])

    # ----------------------------
    # Return the results to DAKOTA
    # ----------------------------

    resultsfile = sys.argv[2]
    write_dakota_results_file(
        resultsfile, resultsdict, paramsdict, active_set_vector)



def parse_dakota_parameters_file(paramsfilename):
    """Return parameters for application."""

    # setup regular expressions for parameter/label matching
    e = r'-?(?:\d+\.?\d*|\.\d+)[eEdD](?:\+|-)?\d+'  # exponential notation
    f = r'-?\d+\.\d*|-?\.\d+'                       # floating point
    i = r'-?\d+'                                    # integer
    value = e + '|' + f + '|' + i                           # numeric field
    tag = r'\w+(?::\w+)*'                           # text tag field

    # regular expression for standard parameters format
    standard_regex = re.compile('^\s*(' + value + ')\s+(' + tag + ')$')

    # open DAKOTA parameters file for reading
    paramsfile = open(paramsfilename, 'r')

    # extract the parameters from the file and store in a dictionary
    paramsdict = {}
    for line in paramsfile:
        m = standard_regex.match(line)
        if m:
            # print m.group()
            paramsdict[m.group(2)] = m.group(1)

    paramsfile.close()

    return paramsdict


def write_dakota_results_file(
        resultfilename, resultsdict, paramsdict, active_set_vector):
    """Write results of application for Dakota."""

    # Make sure number of functions is as expected.
    num_fns = 0
    if ('functions' in paramsdict):
        num_fns = int(paramsdict['functions'])
    if num_fns != len(resultsdict['fns']):
        raise Exception('Number of functions not as expected.')

    # write outputfile
    outfile = open(resultfilename, 'w')

    for func_ind in range(0, num_fns):
        # write functions
        if (active_set_vector[func_ind] & 1):
            functions = resultsdict['fns']
            outfile.write(str(functions[func_ind]) +
                          ' f' + str(func_ind) + '\n')

    # write gradients
    for func_ind in range(0, num_fns):
        if (active_set_vector[func_ind] & 2):
            grad = resultsdict['fnGrads'][func_ind]
            outfile.write('[ ')
            for deriv in grad:
                outfile.write(str(deriv) + ' ')
            outfile.write(']\n')

    outfile.close()

def check_nvar(nvar,paramsdict):
    """Check to make sure we have the right number of uncertain variables."""
    num_vars = 0
    if ('variables' in paramsdict):
        num_vars = int(paramsdict['variables'])
    if (num_vars != nvar):
        #print 'Error: Simulation expected ' + str(nvar) + ' variables, found ' \
        #    + str(num_vars) + ' variables.'
        #sys.exit()
        message = 'Error: Simulation expected ' + str(nvar) + ' variables, found ' \
            + str(num_vars) + ' variables.'
        raise Exception(message)




if __name__ == '__main__':
    main()
