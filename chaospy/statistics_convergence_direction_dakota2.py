
import numpy as np
# import matplotlib.pyplot as plt
import json
from getSamplePoints import getSamplePoints
from windfarm_set_up import problem_set_up
from dakotaInterface import updateDakotaFile, updateDakotaFile3
from wind_pdfs import wind_speed_pdfweibull, wind_direction_pdf


# Put this in the simple case
# ------ Problem we are solving is of the form ------ #
#   mu_p = int p(x)rho(x) dx = sum p(x_i)rho(x_i)w(x_i)
#   where p is the power, x is the uncertain variable
#   rho is the probability density function of the
#   uncertain variable and w is the numerical
#   integration weight.
# --------------------------------------------------- #


if __name__ == "__main__":

    mean = []
    std = []
    samples = []
    for n in range(10,11,1):

        # n = 10  # Number of points
        dakotaFileName = 'dakotaAEPdirection2.in'

        # Update dakota file with desired number of sample points
        updateDakotaFile3(dakotaFileName, n)

        # run Dakota file to get the points locations (also weights)
        points = getSamplePoints(dakotaFileName)

        # For wind speed
        windspeeds = np.ones(n)*8
        winddirections = points

        print 'Locations at which power is evaluated'
        for point in winddirections:
            print '\t', point


        # Set up problem, define the turbine locations and all that stuff, pass it the wind direction x
        prob = problem_set_up(windspeeds, winddirections, dakotaFileName)

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
    print np.mean(power)*hours/1e6
    print np.std(power)*hours/1e6

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
    # ax.plot(winddirections, power)
    # ax.set_xlabel('wind speed (m/s)')
    # ax.set_ylabel('power')

    # fig, ax = plt.subplots()
    # ax.plot(samples,mean)
    # ax.set_xlabel('Number of Wind Directions')
    # ax.set_ylabel('mean annual energy production')
    # ax.set_title('Mean annual energy as a function of the Number of Wind Directions')

    # plt.show()


