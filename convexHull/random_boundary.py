"""
Description:    Generates random, convex wind farm boundary vertices
Date:           5/19/2016
Author:         Jared J. Thomas
"""

from wakeexchange.GeneralWindFarmComponents import calculate_boundary, calculate_distance

import numpy as np
import matplotlib.pyplot as plt


def create_vertices(low=0.0, high=1000.0, npoints=100):

    if npoints < 3:
        npoints += 3

    # np.random.seed(64)

    points = (np.random.rand(npoints, 2)+low)*high

    print points

    boundaryVertices, boundaryNormals = calculate_boundary(points)

    return boundaryVertices


def rotate(A, theta):

    rotatation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta), np.cos(theta)]])

    B = np.dot(A, rotatation_matrix)

    return B


def shift_to_positive(A):

    for i in range(0, len(A.shape)):
        if min(A[:, i]) < 0:
            A[:, i] -= min(A[:, i])
    print A
    return A


def shift_to_zero(A):
    for i in range(0, len(A.shape)):
        A[:, i] -= min(A[:, i])
    return A

if __name__ == '__main__':

    vertices = create_vertices(high=7*8*126.4, npoints=int(np.random.random()*20))
    vertices = rotate(vertices, np.random.random()*np.pi*2.0)
    vertices = shift_to_positive(vertices)
    vertices = shift_to_zero(vertices)

    np.savetxt('outputfiles/random_boundary.txt', np.c_[vertices[:, 0], vertices[:, 1]])

    plt.plot(vertices[:, 0], vertices[:, 1], 'k')
    plt.plot([vertices[0, 0], vertices[-1, 0]], [vertices[0, 1], vertices[-1, 1]], 'k')
    plt.show()