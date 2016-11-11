
# This is the case for the direct python interface.

def pythonInterface(**kwargs):

    # import numpy as np  # It doesn't like it because dakota and my python numpy are inconsistent
    # http://stackoverflow.com/questions/35006614/what-does-symbol-not-found-expected-in-flat-namespace-actually-mean
    # The code inside the try statement doesn't use numpy to read the file anymore.

    paramsdict = kwargs

    num_fns = int(paramsdict['functions'])

    active_set_vector = []
    for i in range(num_fns):
        active_set_vector.append(int(paramsdict['asv'][i]))

    try:
        f = open('powerInput.txt')
        f.readline()  # Skip the header
        lines = f.readlines()
        index = int(paramsdict['currEvalId']) - 1
        power_i = float(lines[index])

    except IOError:  # This is for the case when we are only getting the sample points.
        print '\n\nWARNING: missing powerInput.txt\n\n'
        index = int(paramsdict['currEvalId']) - 1
        power_i = index  # -1.0  # np.nan

    resultsdict = {'fns': [power_i], 'fnGrads': []}

    return resultsdict
