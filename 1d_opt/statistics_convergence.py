
import numpy as np
import matplotlib.pyplot as plt
import json
from openmdao.api import Problem
from AEPGroups import AEPGroup
import distributions
import windfarm_setup


def run():
    """
    method_dict = {}
    keys of method_dict:
        'method' = 'dakota', 'rect' or 'chaospy'  # 'chaospy needs updating
        'uncertain_var' = 'speed' or 'direction'
        'layout' = 'amalia', 'optimized', 'grid', 'random', 'lhs'
        'distribution' = a distribution object
        'dakota_filename' = 'dakotaInput.in', applicable for dakota method
    Returns:
        Writes a json file 'record.json' with the run information.
    """

    method_dict = {}
    method_dict['method']           = 'dakota'
    method_dict['uncertain_var']    = 'direction'
    method_dict['layout']           = 'optimized'

    if method_dict['uncertain_var'] == 'speed':
        dist = distributions.getWeibull()
        method_dict['distribution'] = dist
    elif method_dict['uncertain_var'] == 'direction':
        dist = distributions.getWindRose()
        method_dict['distribution'] = dist
    else:
        raise ValueError('unknown uncertain_var option "%s", valid options "speed" or "direction".' %method_dict['uncertain_var'])

    method_dict['dakota_filename'] = 'dakotageneral.in'

    mean = []
    std = []
    samples = []

    for n in range(5,6,1):

        ### Set up the wind speeds and wind directions for the problem ###

        points, weights = windfarm_setup.getPoints(method_dict, n)

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


        # Turbines layout
        turbineX, turbineY = windfarm_setup.getLayout(method_dict['layout'])

        # turbine size and operating conditions

        rotor_diameter = 126.4  # (m)
        air_density = 1.1716    # kg/m^3

        # initialize arrays for each turbine properties
        nTurbs = turbineX.size
        rotorDiameter = np.zeros(nTurbs)
        axialInduction = np.zeros(nTurbs)
        Ct = np.zeros(nTurbs)
        Cp = np.zeros(nTurbs)
        generator_efficiency = np.zeros(nTurbs)
        yaw = np.zeros(nTurbs)

        # define initial values
        for turbI in range(nTurbs):
            rotorDiameter[turbI] = rotor_diameter
            axialInduction[turbI] = 1.0/3.0
            Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
            Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
            generator_efficiency[turbI] = 0.944
            yaw[turbI] = 0.     # deg.

        # initialize problem
        prob = Problem(AEPGroup(nTurbines=nTurbs, nDirections=n,
                                method_dict=method_dict))
        prob.setup(check=False)

        # assign initial values to variables
        prob['windSpeeds'] = windspeeds
        prob['windDirections'] = winddirections
        prob['weights'] = weights
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generator_efficiency
        prob['air_density'] = air_density
        prob['Ct_in'] = Ct
        prob['Cp_in'] = Cp

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        for direction_id in range(0, n):
            prob['yaw%i' % direction_id] = yaw

        # Run the problem
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
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
           'layout': method_dict['layout']}
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
    run()
    # plot()


