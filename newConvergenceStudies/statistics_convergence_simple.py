
import numpy as np
import matplotlib.pyplot as plt
import json
from windfarm_set_up import problem_set_up
from wind_pdfs import wind_speed_pdfweibull, wind_direction_pdf


if __name__ == "__main__":

    AEP = []
    std_energy = []
    samples = []
    for n in range(100,101,1):

        # ------- Define the winddirections and windspeeds vector
        #         and the associated integration weights --------

        # For wind direction
        a = 0.0
        b = 360.0
        step = (b-a)/n
        winddirections = np.linspace(step/2, b-step/2, n)
        weights = np.ones(n)*step  # Integration weight, the dx interval
        windspeeds = np.ones(n)*8
        rho = wind_direction_pdf(winddirections)
        # rho = np.ones(n)/(b-a)  #  The uniform density. # Note this different than the dakota case, because dakota does the weighting. This is kinda the opposite of factor.



        # For wind speed
        # a = 0.0
        # b = 30.0
        # step = (b-a)/n
        # windspeeds = np.linspace(step/2,b-step/2,n)
        # weights = np.ones(n)*step  # Integration weight, the dx interval
        # winddirections = np.ones(n)*225
        # rho = wind_speed_pdfweibull(windspeeds)
        # # rho = np.ones(n)/(b-a)  #  The uniform density. # Note this different than the dakota case, because dakota does the weighting.

        print 'Locations at which power is evaluated'
        for point in winddirections:
            print '\t', point

        # Set up problem, define the turbine locations and all that stuff
        prob = problem_set_up(windspeeds, winddirections, weights, rho)

        prob.run()

        # print the results
        AEP_data = prob['AEP']
        std_energy_data = prob['std_energy']
        print 'AEP = ', AEP_data/1e6, ' GWhrs'
        print 'std energy = ', std_energy_data/1e6, ' GWhrs'
        AEP.append(AEP_data/1e6)
        std_energy.append(std_energy_data/1e6)
        samples.append(n)



    # Save a record of the run
    power = prob['dirPowers']
    power = power*rho  # Saved the weighted power
    obj = {'AEP': AEP, 'std_energy': std_energy, 'samples': samples, 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'dirPowers': power.tolist()}
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

    plt.show()


