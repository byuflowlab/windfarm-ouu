
import numpy as np
# import matplotlib.pyplot as plt
import json
import chaospy as cp
from getSamplePoints import getSamplePoints
from windfarm_set_up import problem_set_up
from dakotaInterface import updateDakotaFile, updateDakotaFile2
# from wind_pdfs import wind_speed_pdfweibull, wind_direction_pdf
import distributions

# Put this in the simple case
# ------ Problem we are solving is of the form ------ #
#   mu_p = int p(x)rho(x) dx = sum p(x_i)rho(x_i)w(x_i)
#   where p is the power, x is the uncertain variable
#   rho is the probability density function of the
#   uncertain variable and w is the numerical
#   integration weight.
# --------------------------------------------------- #


if __name__ == "__main__":

    method = 'chaospy'
    mean = []
    std = []
    samples = []

    # Set up the distribution
    # windrose_dist = distributions.getWindRose()
    weibull_dist = distributions.getWeibull()
    # weibull_dist = cp.weibull(a=0.1)
    # weibull_dist = distributions.getWindRose()


    for n in range(20,21,1):

        # I need to get the points and weights
        points, weights = cp.generate_quadrature(order=n-1, domain=weibull_dist, rule="Clenshaw")
        # points, weights = cp.generate_quadrature(order=n, domain=weibull_dist, rule="L")
        # points, weights = distributions.rectangle(n)

        windspeeds = points[0]
        winddirections = np.ones(n)*225

        # windspeeds = np.ones(n)*8
        # winddirections = points[0]

        print 'Locations at which power is evaluated'
        print '\twindspeed \t winddirection'
        for i in range(n):
            print i+1,'\t', '%.2f' %windspeeds[i], '\t', '%.2f' %winddirections[i]

        # Set up problem, define the turbine locations and all that stuff, pass it the wind direction x
        prob = problem_set_up(windspeeds, winddirections, method)

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


