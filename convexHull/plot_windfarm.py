"""
Description:    Plot a given wind farm layout
Date:           5/19/2016
Author:         Jared J. Thomas
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_windfarm_layout(vertices, turbines, rotor_diameter, title="", filename=""):

    linewidth = 1

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 16}

    plt.rc('font', **font)

    plt.rcParams['xtick.major.pad'] = '15'
    plt.rcParams['ytick.major.pad'] = '15'


    fig = plt.figure()
    ax = fig.add_axes([0., 0.2, 1.0, 0.6])
    ax.set_aspect('equal')



    # plot turbine locations
    for i in range(0, np.size(turbines[:, 0])):
        circle = plt.Circle((turbines[i, 0]/rotor_diameter, turbines[i, 1]/rotor_diameter), 0.5, color='k', clip_on=False,
                            fill=None, linestyle='-',
                        linewidth=linewidth)
        ax.add_artist(circle)

    # plot boundary
    plt.plot(vertices[:, 0]/rotor_diameter, vertices[:, 1]/rotor_diameter, '--k')
    plt.plot(np.array([vertices[0, 0], vertices[-1, 0]])/rotor_diameter, np.array([vertices[0, 1], vertices[-1, 1]])/rotor_diameter, '--k')

    plt.xlabel('X position ($X/D_r$)')
    plt.ylabel('Y position ($Y/D_r$)')
    plt.title(title)

    if filename is not "":
        plt.savefig('outputfiles/' + filename + '.pdf')

    plt.show()

    return

if __name__ == "__main__":

    boundary = 0
    layout = 0

    for layout in range(0, 10):

        rotor_diameter = 126.4

        vertices_file = 'outputfiles/random_boundary%i.txt' % boundary
        layout_file = 'outputfiles/layout%i%i.txt' % (boundary, layout)

        vertices = np.loadtxt(vertices_file)
        locations = np.loadtxt(layout_file)

        print vertices, locations

        plot_windfarm_layout(vertices, locations, rotor_diameter, title="Boundary %i, Layout %i" % (boundary, layout),
                             filename="windfarm%i%i" % (boundary, layout))