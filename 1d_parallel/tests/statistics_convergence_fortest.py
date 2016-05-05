
import numpy as np
# import matplotlib.pyplot as plt
import json
from getSamplePoints import getSamplePoints
from windfarm_set_up import problem_set_up
from dakotaInterface import updateDakotaFile, updateDakotaFile2
import distributions
import quadrature_rules


def getPoints(method_dict, n):

    method = method_dict['method']
    if method == 'dakota':
        # Update dakota file with desired number of sample points
        updateDakotaFile(method_dict['dakota_filename'], n)
        # run Dakota file to get the points locations
        points = getSamplePoints(method_dict['dakota_filename'])

        # If direction
        # Update the points to correct range
        # a = 140
        # b = 470
        # points = ((b+a)/2. + (b-a)*points)%360
        return points

    if method == 'rect':

        if method_dict['uncertain_var'] == 'speed':
            dist = distributions.getWeibull()
            method_dict['distribution'] = dist
        elif method_dict['uncertain_var'] == 'direction':
            dist = distributions.getWindRose()
            method_dict['distribution'] = dist
        else:
            raise ValueError('unknown uncertain_var option "%s", valid options "speed" or "direction".' %method_dict['uncertain_var'])

        points, unused = quadrature_rules.rectangle(n, method_dict['distribution'])
        return points[0]


def run(method_dict, n):
    """
    method_dict = {}
    keys of method_dict:
        'method' = 'dakota', 'rect' or 'chaospy'  # 'chaospy needs updating
        'uncertain_var' = 'speed' or 'direction'
        'dakota_filename' = 'dakotaInput.in', applicable for dakota method
        'distribution' = a distribution applicable for rect and chaospy methods
    Returns:
        Writes a json file 'record.json' with the run information.
    """

    mean = []
    std = []
    samples = []

    points = getPoints(method_dict, n)

    if method_dict['uncertain_var'] == 'speed':
        # For wind speed
        windspeeds = points
        winddirections = np.ones(n)*225
    elif method_dict['uncertain_var'] == 'direction':
        # For wind direction
        windspeeds = np.ones(n)*8
        winddirections = points
    else:
        raise ValueError('unknown uncertain_var option "%s", valid options "speed" or "direction".' %method_dict['uncertain_var'])


    print 'Locations at which power is evaluated'
    print '\twindspeed \t winddirection'
    for i in range(n):
        print i+1, '\t', '%.2f' % windspeeds[i], '\t', '%.2f' % winddirections[i]

    # Set up problem, define the turbine locations and all that stuff
    prob = problem_set_up(windspeeds, winddirections, method_dict)

    prob.run()

    # print the results
    mean_data = prob['mean']
    std_data = prob['std']
    print 'mean = ', mean_data/1e6, ' GWhrs'
    print 'std = ', std_data/1e6, ' GWhrs'
    mean.append(mean_data/1e6)
    std.append(std_data/1e6)
    samples.append(n)


    # Save a record of the run
    power = prob['power']

    obj = {'mean': mean, 'std': std, 'samples': samples, 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': power.tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var']}
    jsonfile = open('record.json','w')
    json.dump(obj, jsonfile, indent=2)
    jsonfile.close()


def plot():
    jsonfile = open('record.json','r')
    a = json.load(jsonfile)
    print a
    print type(a)
    print a.keys()
    print json.dumps(a, indent=2)
    jsonfile.close()

    # fig, ax = plt.subplots()
    # ax.plot(windspeeds, power)
    # ax.set_xlabel('wind speed (m/s)')
    # ax.set_ylabel('power')
    #
    # fig, ax = plt.subplots()
    # ax.plot(samples,mean)
    # ax.set_xlabel('Number of Wind Directions')
    # ax.set_ylabel('mean annual energy production')
    # ax.set_title('Mean annual energy as a function of the Number of Wind Directions')
    #
    # plt.show()


if __name__ == "__main__":
    method_dict = {}
    method_dict['method'] = 'rect'
    method_dict['uncertain_var'] = 'speed'

    method_dict['dakota_filename'] = 'dakotaAEPspeed.in'
    # method_dict['dakota_filename'] = 'dakotaAEPdirection.in'
    # method_dict['dakota_filename'] = 'dakotadirectionsmooth.in'
    run(method_dict, 10)
    # plot()


