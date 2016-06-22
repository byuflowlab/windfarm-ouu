"""
Description:    Modified the random_layout.py file to give an estimate of the area of the convex hull with respect
                to the containing rectangle.
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
    inside = 0
    for i in range(0, npoints):

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
            inside = inside + 1

    return inside


if __name__ == '__main__':

    # vertices = np.loadtxt('outputfiles/random_boundary.txt')
    vertices = np.loadtxt('../WindFarms/layout_amalia.txt')
    N = 1000000
    inside_points = generate_layout(vertices, npoints=N)
    # Estimate of the area points inside over all points
    print 'Area estimate = ', inside_points/float(N)
    # with N = 1million the estimate area was 0.782363 and 0.781395 and one more time 0.781356. Take 0.78 of area, scaling the grid case by this factor still produces slightly more energy than the Amalia. without scaling AEP 756 with scaling 726 or so. Don't do any updating of the results.
    # I can reduce each side of the rectangle by 0.884