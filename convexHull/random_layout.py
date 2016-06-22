"""
Description:    Generates random wind farm turbine layout within a given convex boundary
Date:           5/19/2016
Author:         Jared J. Thomas
"""

from florisse.GeneralWindFarmComponents import calculate_distance, calculate_boundary

import numpy as np
import matplotlib.pyplot as plt
from time import sleep


def generate_layout(vertices, npoints=60):

    # get unit normals
    boundaryVertices, unit_normals = calculate_boundary(vertices)
    print unit_normals

    # initialize points array
    points = np.zeros([npoints, 2])

    # deterine high and low values for containing rectangle
    highx = max(vertices[:, 0])
    lowx = min(vertices[:, 0])
    highy = max(vertices[:, 1])
    lowy = min(vertices[:, 1])

    print highx
    print lowx
    print highy
    print lowy

    # generate random points within the wind farm boundary
    np.random.seed(101)
    for i in range(0, npoints):

        good_point = False

        while not good_point:

            # generate random point in containing rectangle
            point = np.random.rand(1, 2)
            # print point
            point[:, 0] = point[:, 0]*(highx - lowx) + lowx
            point[:, 1] = point[:, 1]*(highy - lowy) + lowy
            # print point

            # calculate signed distance from the point to each boundary facet
            # distance = calculate_distance(point, vertices, unit_normals)
            distance = calculate_distance(point, boundaryVertices, unit_normals)


            # determine if the point is inside the wind farm boundary
            if all(d >= 0 for d in distance[0]):
                good_point = True
                points[i, :] = point[0, :]

    return points, boundaryVertices


if __name__ == '__main__':

    # vertices = np.loadtxt('outputfiles/random_boundary.txt')
    vertices = np.loadtxt('../WindFarms/layout_amalia.txt')
    print vertices
    print len(vertices)
    points, vertices = generate_layout(vertices, npoints=60)

    np.savetxt('outputfiles/layout.txt', np.c_[points[:, 0], points[:, 1]])

    # plot points
    plt.scatter(points[:, 0], points[:, 1])

    # plot boundary
    plt.plot(vertices[:, 0], vertices[:, 1], 'k')
    plt.plot([vertices[0, 0], vertices[-1, 0]], [vertices[0, 1], vertices[-1, 1]], 'k')

    plt.show()