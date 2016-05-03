
import numpy as np
# import matplotlib.pyplot as plt
import json
from getSamplePoints import getSamplePoints
from windfarm_set_up import problem_set_up
from dakotaInterface import updateDakotaFile, updateDakotaFile2


if __name__ == "__main__":

    method = 'dakota'
    method_dict = {}
    mean = []
    std = []
    samples = []

    method_dict['filename'] = 'dakotaAEPspeed.in'
    # method_dict['filename'] = 'dakotaAEPdirection.in'
    # method_dict['filename'] = 'dakotadirectionsmooth.in'

    for n in range(1,24,1):

        # Update dakota file with desired number of sample points
        updateDakotaFile(method_dict['filename'], n)

        # run Dakota file to get the points locations (also weights)
        points = getSamplePoints(method_dict['filename'])

        # Update the points to correct range
        # a = 140
        # b = 470
        # points = ((b+a)/2. + (b-a)*points)%360

        # For wind speed
        windspeeds = points
        winddirections = np.ones(n)*225

        # For wind direction
        # windspeeds = np.ones(n)*8
        # winddirections = points

        print 'Locations at which power is evaluated'
        print '\twindspeed \t winddirection'
        for i in range(n):
            print i+1, '\t', '%.2f' % windspeeds[i], '\t', '%.2f' % winddirections[i]

        # Set up problem, define the turbine locations and all that stuff, pass it the wind direction x
        prob = problem_set_up(windspeeds, winddirections, method, method_dict)

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
    hours = 8760
    # print np.mean(power)*hours/1e6
    # print np.std(power)*hours/1e6

    obj = {'mean': mean, 'std': std, 'samples': samples, 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': power.tolist()}
    jsonfile = open('record.json','w')
    json.dump(obj, jsonfile, indent=2)
    jsonfile.close()
    # jsonfile = open('record.json','r')
    # a = json.load(jsonfile)
    # print a
    # print type(a)
    # print a.keys()
    # print json.dumps(a, indent=2)

    # fig, ax = plt.subplots()
    # ax.plot(windspeeds, power)
    # ax.set_xlabel('wind speed (m/s)')
    # ax.set_ylabel('power')

    # fig, ax = plt.subplots()
    # ax.plot(samples,mean)
    # ax.set_xlabel('Number of Wind Directions')
    # ax.set_ylabel('mean annual energy production')
    # ax.set_title('Mean annual energy as a function of the Number of Wind Directions')

    # plt.show()


