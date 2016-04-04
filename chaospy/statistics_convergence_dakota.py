
import numpy as np
import matplotlib.pyplot as plt
import json
from getSamplePoints import getSamplePoints
from windfarm_set_up import problem_set_up
from dakotaInterface import updateDakotaFile
from wind_pdfs import wind_speed_pdfweibull, wind_direction_pdf


# ------ Problem we are solving is of the form ------ #
#   mu_p = int p(x)rho(x) dx = sum p(x_i)rho(x_i)w(x_i)
#   where p is the power, x is the uncertain variable
#   rho is the probability density function of the
#   uncertain variable and w is the numerical
#   integration weight.
# --------------------------------------------------- #




if __name__ == "__main__":

    AEP = []
    std_energy = []
    samples = []
    for n in range(20,21,1):

        # n = 10  # Number of points
        dakotaFileName = 'dakotaAEPdirection.in'
        # dakotaFileName = 'dakotaAEPspeed.in'

        # Update dakota file with desired number of sample points
        updateDakotaFile(dakotaFileName, n)

        # run Dakota file to get the points locations (also weights)
        points, weights = getSamplePoints(dakotaFileName)       # The sample points could return 1 or 2D.

        # For wind direction
        windspeeds = np.ones(n)*8
        winddirections = points
        rho = wind_direction_pdf(winddirections)
        # rho = np.ones(n)


        # For wind speed
        # windspeeds = points
        # winddirections = np.ones(n)*225
        # # rho = wind_speed_pdfweibull(windspeeds)
        # rho = np.ones(n)

        print 'Locations at which power is evaluated'
        for point in winddirections:
            print '\t', point


        # Set up problem, define the turbine locations and all that stuff, pass it the wind direction x
        prob = problem_set_up(windspeeds, winddirections, weights, rho, dakotaFileName)

        prob.run()

        # print the results
        factor = 360  # To correct if feeding to dakota weighted response. Equals the range of the uniform variable
        AEP_data = prob['AEP']*factor
        std_energy_data = np.sqrt(prob['Var_energy']/8760.)*factor*8760
        std_energy_data = np.sqrt(prob['Var_energy']/8760.*factor)*8760
        var_energy_data = prob['Var_energy']
        print 'AEP = ', AEP_data/1e6, ' GWhrs'
        print 'std energy = ', std_energy_data/1e6, ' GWhrs'
        AEP.append(AEP_data/1e6)
        std_energy.append(std_energy_data/1e6)
        samples.append(n)


    # Save a record of the run
    power = prob['power']
    power = power*rho  # Saved the weighted power
    obj = {'AEP': AEP, 'std_energy': std_energy, 'samples': samples, 'winddirections': winddirections.tolist(),
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

    fig, ax = plt.subplots()
    ax.plot(winddirections, power)
    ax.set_xlabel('wind speed (m/s)')
    ax.set_ylabel('weighted power')


    # fig, ax = plt.subplots()
    # ax.plot(samples,AEP)
    # ax.set_xlabel('Number of Wind Directions')
    # ax.set_ylabel('AEP')
    # ax.set_title('AEP as a function of the Number of Wind Directions')
    # plt.savefig('convergence.pdf')



    plt.show()


